# C:\Users\nuri\Desktop\RAG\ai-stack\api\main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil, os, logging, re, uuid, json, math
from fastapi.responses import RedirectResponse
from fastapi import Body
import time
import traceback
import src.vectorstore as VS
from pydantic import BaseModel
from datetime import datetime
from zoneinfo import ZoneInfo
import inspect
import unicodedata

from src.utils import proxy_get, call_chat_completions, drop_think
from src.rag_pipeline import build_rag_chain, Document
from src.loaders import load_docs_any
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.ext.confluence_mcp import mcp_search
import asyncio, hashlib
from collections import Counter, defaultdict
from src.retrieval.rerank import parse_query_intent, pick_for_injection
from api.smart_router import router as smart_router
from FlagEmbedding import FlagReranker
import httpx

# empty FAISS 빌드를 위한 보조들
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS as FAISSStore
from langchain_community.docstore.in_memory import InMemoryDocstore

logging.basicConfig(level=logging.INFO)
DISABLE_INTERNAL_MCP = (os.getenv("DISABLE_INTERNAL_MCP", "0").lower() in ("1","true","yes"))
MCP_WRITEBACK = (os.getenv("MCP_WRITEBACK", "off").lower() in ("1","true","yes","on","persist"))
MCP_WRITEBACK_TITLES_ONLY = (os.getenv("MCP_WRITEBACK_TITLES_ONLY","off").lower() in ("1","true","yes","on"))

PAGE_FILTER_MODE = getattr(settings, "PAGE_FILTER_MODE", "soft")   # "soft"|"hard"|"off"
PAGE_HINT_BONUS = float(getattr(settings, "PAGE_HINT_BONUS", 0.35))
SINGLE_SOURCE_COALESCE = bool(getattr(settings, "SINGLE_SOURCE_COALESCE", True))
COALESCE_THRESHOLD = float(getattr(settings, "COALESCE_THRESHOLD", 0.55))
_COALESCE_TRIGGER = re.compile(r"(최근|이슈|목록|정리|요약|top\s*\d+|\d+\s*가지)", re.I)

SPACE_FILTER_MODE = getattr(settings, "SPACE_FILTER_MODE", "hard")
SPACE_HINT_BONUS = float(getattr(settings, "SPACE_HINT_BONUS", 0.25))
TITLE_BONUS      = float(getattr(settings, "TITLE_BONUS", 0.30))
ENABLE_SPARSE    = bool(getattr(settings, "ENABLE_SPARSE", False))
SPARSE_LIMIT     = int(getattr(settings, "SPARSE_LIMIT", 150))

MAX_FALLBACK_SECS = int(os.getenv("MAX_FALLBACK_SECS", "7"))   # 폴백 전체 상한(초)
MCP_TIMEOUT       = int(os.getenv("MCP_TIMEOUT", "5"))         # 각 MCP 콜 타임아웃(초)
MCP_MAX_TASKS     = int(os.getenv("MCP_MAX_TASKS", "4"))       # 동시 질의 최대 개수

TZ_NAME = getattr(settings, "TZ_NAME", "Asia/Seoul")

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")
MCP_HTTP_BASE_URL = os.getenv("MCP_HTTP_BASE_URL", "").rstrip("/")

LOCAL_FIRST = bool(getattr(settings, "LOCAL_FIRST", True))
LOCAL_BONUS = float(getattr(settings, "LOCAL_BONUS", 0.25))

META_PAT  = re.compile(r"(주제|개요|요약|무엇|무슨\s*내용)", re.I)
TITLE_PAT = re.compile(r"(제목|title|문서명|파일명)", re.I)
LOGIN_PAT = re.compile(r"(Confluence에\s*로그인|로그인\s*-\s*Confluence|name=[\"']os_username[\"'])", re.I)
THIS_FILE_PAT = re.compile(
    r"(이\s*(?:pdf|엑셀|xlsx|csv|txt|md|hwp|한글|워드|ppt|pptx|파워포인트)?\s*(?:파일|문서|자료)|"
    r"해당\s*(?:파일|문서|자료|pdf|엑셀|pptx|csv|txt|md)|"
    r"첨부(?:한)?\s*(?:파일|문서|자료|pdf|엑셀|pptx|csv|txt|md)|"
    r"방금\s*(?:올린|업로드한)\s*(?:파일|문서|자료|pdf|엑셀|pptx|csv|txt|md)|"
    r"위\s*(?:파일|문서|자료))",
    re.I
)
_FILE_EXT_RE = re.compile(r"\.(pdf|pptx|ppt|xlsx|xls|csv|txt|md|docx)($|\?)", re.I)
_CHAPTER_RE = re.compile(r"제\s*(\d+)\s*장")
_ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조")

# === 세션 기반 Sticky 관리 (경쟁 조건 해결) ===
class SessionManager:
    """세션별 sticky 소스 관리 - 동시 사용자 간섭 방지"""
    def __init__(self):
        self._sessions = {}  # session_id -> {"source": str, "until": float, "last_source": str}
        self._lock = asyncio.Lock()
        self._cleanup_interval = 300  # 5분마다 만료된 세션 정리

    async def get_sticky(self, session_id: Optional[str]) -> Optional[str]:
        """세션의 현재 sticky 소스 반환"""
        if not session_id:
            return None
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess and time.time() < sess.get("until", 0):
                return sess.get("source")
            return None

    async def set_sticky(self, session_id: Optional[str], source: str, duration: int = 180):
        """세션의 sticky 소스 설정"""
        if not session_id:
            return
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {}
            self._sessions[session_id]["source"] = _norm_source(source)
            self._sessions[session_id]["until"] = time.time() + duration
            self._sessions[session_id]["last_source"] = _norm_source(source)

    async def clear_sticky(self, session_id: Optional[str]):
        """세션의 sticky 소스 해제"""
        if not session_id:
            return
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["source"] = None
                self._sessions[session_id]["until"] = 0

    async def get_last_source(self, session_id: Optional[str]) -> Optional[str]:
        """세션의 마지막 업로드/인덱싱 소스 반환"""
        if not session_id:
            return None
        async with self._lock:
            sess = self._sessions.get(session_id)
            return sess.get("last_source") if sess else None

    async def cleanup_expired(self):
        """만료된 세션 정리"""
        async with self._lock:
            now = time.time()
            expired = [sid for sid, sess in self._sessions.items()
                      if sess.get("until", 0) < now - 600]  # 10분 이상 지난 것만
            for sid in expired:
                del self._sessions[sid]

session_manager = SessionManager()

# 후방 호환성: 전역 sticky (session_id 없는 요청용)
last_source: Optional[str] = None
current_source: Optional[str] = None
current_source_until: float = 0.0
STICKY_SECS = int(getattr(settings, "STICKY_SECS", 180))

STICKY_STRICT = bool(getattr(settings, "STICKY_STRICT", True))
STICKY_FROM_COALESCE = bool(getattr(settings, "STICKY_FROM_COALESCE", False))
STICKY_AFTER_MCP = bool(getattr(settings, "STICKY_AFTER_MCP", True))         

# Sticky 동작 모드/가중치/소스 쏠림 상한 (기본: 기존과 동일)
STICKY_MODE   = str(getattr(settings, "STICKY_MODE", "bonus")).lower()   # filter = pdf에 집착, bonus = 발란스 형
STICKY_BONUS  = float(getattr(settings, "STICKY_BONUS", 0.18))           # bonus 모드에서 가산점
PER_SOURCE_CAP = int(getattr(settings, "PER_SOURCE_CAP", 0))             # 0=off, >0이면 소스당 최대 N개

SHOW_SOURCE_BLOCK = os.getenv("SHOW_SOURCE_BLOCK", "auto").lower()  # auto|never|always

# ← [ADD] 앵커 추출(질문 핵심어) + sticky 유효성 검사
_GENERIC = set("보고 보고서 리포트 정보 정리 페이지 자료 문서 요약 정책 통계 항목 사이트 url 링크 출처".split())

# >>> [ADD] pageId 추출/URL 정규화 유틸
_PAGEID_RE = re.compile(r"[?&]pageId=(\d+)")


# 1) '오늘/지금/현재' 같은 지시어가 필수
_DEICTIC_RE = re.compile(r"(오늘|지금|현재)", re.I)

# 2) 사용자가 특정 날짜를 콕 집은 경우는 RAG로(예: 2025년 7월 3일)
_EXPLICIT_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월\s*\d{1,2}\s*일", re.I)

# 3) 리스트/요약/탑N 같은 RAG 냄새 신호가 있으면 RAG로
_RETRIEVAL_HINT_RE = re.compile(r"(최근|이슈|목록|리스트|정리|요약|top\s*\d+|\d+\s*가지|내역|중)", re.I)

# 4) 날짜/시간을 '오늘' 기준으로 물어보는지
_DATE_TIME_NEED_RE = re.compile(r"(날짜|요일|시간|시각|몇\s*시|몇\s*분|몇\s*월|몇\s*일|며칠)", re.I)

# === v1_chat 출처 호스트 화이트리스트(환경변수) ===
SOURCE_HOST_WHITELIST = [h.strip().lower() for h in os.getenv("ALLOWED_SOURCE_HOSTS","").split(",") if h.strip()]
_URL_HOST_RE  = re.compile(r"^https?://([^/]+)")

ACRONYM_TITLE_BONUS = float(getattr(settings, "ACRONYM_TITLE_BONUS", 0.45))
ACRONYM_BODY_BONUS  = float(getattr(settings, "ACRONYM_BODY_BONUS", 0.25))

# --- Dynamic domain/space signals (no hardcoded lists) ---
DOMAIN_PURITY_THRESHOLD = float(os.getenv("DOMAIN_PURITY_THRESHOLD", "0.6"))
DOMAIN_MIN_STRONG_TOKENS = int(os.getenv("DOMAIN_MIN_STRONG_TOKENS", "1"))
SPACE_SCORE_MIN = int(os.getenv("SPACE_SCORE_MIN", "2"))
MCP_FALLBACK_ON_EMPTY = (os.getenv("MCP_FALLBACK_ON_EMPTY", "1").lower() not in ("0","false","no"))
RAG_SCORE_GATES = (os.getenv("RAG_SCORE_GATES", "0").lower() not in ("0","false","no"))
RAG_TOP1_MIN = float(os.getenv("RAG_TOP1_MIN", "0.45"))
RAG_TOP_GAP_MIN = float(os.getenv("RAG_TOP_GAP_MIN", "0.05"))
RAG_CORE_COVER_MIN = float(os.getenv("RAG_CORE_COVER_MIN", "0.2"))

_DOMAIN_STATS = {
    "space_token_counts": defaultdict(Counter),  # space -> token -> count
    "global_token_counts": Counter(),            # token -> total count
    "purity": {},                                # token -> max(space_count)/total
}

_BAD_TITLE_RE = re.compile(r"(회의록|스크럼|stand[-\s]?up|데일리|일일\s*회의|주간\s*회의|스프린트\s*플래닝|미팅\s*노트)", re.I)

CONFL_HINT_WORDS = [w.strip() for w in os.getenv("CONFL_HINT_WORDS","컨플루언스,컨플,confluence").split(",") if w.strip()]

CITATION_STRIP = os.environ.get("CITATION_STRIP", "auto").lower()

_URL_RE = re.compile(r'(?im)^\s*(?:[-*•]\s*)?(?:https?://\S+)\s*$')
_CITE_HEAD_RE = re.compile(r'(?im)^\s*(?:[-*•]\s*)?(출처|참고|참조|reference|references|sources?)\s*:?.*$')

def _all_sources_are_uploads(sources):
    return bool(sources) and all(str(s).startswith("uploads/") for s in sources)

def _user_requested_citation(q: str) -> bool:
    ql = (q or "").lower()
    return any(x in ql for x in ["출처", "근거", "링크", "url", "레퍼런스", "reference"])

def _strip_citation_block(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out, skipping = [], False
    for line in lines:
        if _CITE_HEAD_RE.match(line):
            skipping = True
            continue
        if skipping:
            # cite 블록이 끝날 때까지 URL/불릿/빈줄은 건너뜀
            if line.strip() == "" or _URL_RE.match(line) or line.strip().startswith(("-", "*", "•")):
                continue
            # 다른 정상 본문 줄이 나오면 블록 종료
            skipping = False
        if not skipping:
            out.append(line)
    # 꼬리의 단독 URL/빈줄/인용헤더 정리
    while out and (_URL_RE.match(out[-1]) or out[-1].strip() == "" or _CITE_HEAD_RE.match(out[-1])):
        out.pop()
    return "\n".join(out).strip()

def _maybe_strip_citations(answer: str, q: str, sources) -> str:
    mode = CITATION_STRIP  # 'auto' | 'always' | 'off'
    if mode == "off":
        return answer
    cond = (mode == "always") or (mode == "auto" and _all_sources_are_uploads(sources) and not _user_requested_citation(q))
    if cond:
        cleaned = _strip_citation_block(answer)
        if cleaned != answer:
            import logging
            logging.getLogger("api.main").info("citation_strip: applied (mode=%s)", mode)
        return cleaned
    return answer

# 언어 강제 공통 래퍼
REPLY_LANG = os.getenv("REPLY_LANG", "ko").lower()

def lang_wrap(s: str) -> str:
    if REPLY_LANG.startswith("ko"):
        return s + "\n\n항상 한국어로 답하세요. 영어로 장황하게 설명하지 마세요."
    return s + "\n\nAlways answer in English."

# 본문 내 약어(강한 도메인 신호) 감지용
_ACRONYM_BLOB_RE = re.compile(r"\b[A-Z]{2,10}\b")

def rebuild_domain_stats_from_index():
    """
    현재 벡터 인덱스의 문서 title/space를 훑어 토큰 통계를 만든다.
    - 하드코딩 없이 '인덱스가 가진 분포'로 도메인/스페이스 힌트를 추론
    """
    global _DOMAIN_STATS
    stats = {
        "space_token_counts": defaultdict(Counter),
        "global_token_counts": Counter(),
        "purity": {},
    }
    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
    except Exception:
        dct = {}

    for d in dct.values():
        md = getattr(d, "metadata", {}) or {}
        title = str(md.get("title", "") or "")
        space = str(md.get("space", "") or "")
        if not (title and space):
            continue
        toks = [t for t in _tokenize_query(_basic_normalize(title))
                if len(t) >= 2 and t not in _K_STOP]
        for t in set(toks):  # DF 비슷하게
            stats["space_token_counts"][space][t] += 1
            stats["global_token_counts"][t] += 1

    purity = {}
    for t, tot in stats["global_token_counts"].items():
        if tot <= 0:
            continue
        max_in_one_space = max((stats["space_token_counts"][s][t]
                                for s in stats["space_token_counts"]), default=0)
        purity[t] = max_in_one_space / float(tot)
    stats["purity"] = purity
    _DOMAIN_STATS = stats

    global CANON_MAP
    if not os.getenv("CANON_MAP_JSON"):  # 환경에서 고정하지 않은 경우만
        CANON_MAP = derive_canon_map_from_index()

def _domainish_dynamic(q: str) -> bool:
    """
    질의가 내부 도메인(사내 문서) 냄새가 나는지 동적으로 판별:
    - 대문자 약어가 있으면 즉시 True
    - 아니면 질의 토큰 중 space-편중(purity)이 높은 토큰이 일정 개수 이상 있으면 True
    """
    if _ACRONYM_BLOB_RE.search(q or ""):
        return True
    toks = [t for t in _tokenize_query(_basic_normalize(q))
            if len(t) >= 2 and t not in _K_STOP]
    pur = _DOMAIN_STATS.get("purity", {})
    strong = [t for t in toks if pur.get(t, 0.0) >= DOMAIN_PURITY_THRESHOLD]
    return len(strong) >= DOMAIN_MIN_STRONG_TOKENS

def _rank_spaces_for_query(q: str) -> list[str]:
    """
    질의 토큰을 기반으로, 인덱스에 기록된 title-토큰 통계와 겹치는 정도로 space를 점수화.
    """
    toks = [t for t in _tokenize_query(_basic_normalize(q))
            if len(t) >= 2 and t not in _K_STOP]
    stc = _DOMAIN_STATS.get("space_token_counts", {})
    scores = Counter()
    for s, counter in stc.items():
        for t in toks:
            if t in counter:
                scores[s] += counter[t]
    ranked = [s for s, sc in scores.most_common() if sc >= SPACE_SCORE_MIN]
    return ranked


# 상단 유틸/정규식 근처
_A_HREF_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I|re.S)
_RAW_URL_RE = re.compile(r'https?://[^\s"\']+pageId=\d+[^\s"\']*', re.I)

# === PDF/업로드 컨텍스트 차단 토글 & 필터 유틸 ===
ALLOW_PDF_CONTEXT = os.getenv("ALLOW_PDF_CONTEXT","0").lower() not in ("0","false","no")
UPLOAD_BLOCK_PREFIX = os.getenv("UPLOAD_BLOCK_PREFIX","uploads/").strip()
_PDF_EXT_RE = re.compile(r"\.pdf($|\?)", re.I)

def _is_pdf_mime(m):
    return bool(m and str(m).lower().startswith("application/pdf"))

def _is_meeting_like(title: str, body: str = "") -> bool:
    s = (title + " " + body[:400]).lower()
    date_hit = bool(re.search(r"\b20\d{2}[-./]\d{1,2}[-./]\d{1,2}\b", s))
    attendee_hit = any(k in s for k in ("참석자","회의 일시","agenda","minutes"))
    return date_hit and attendee_hit

def _looks_blocked_source(u):
    if not u: 
        return False
    s = str(u).strip().replace("\\","/")
    if UPLOAD_BLOCK_PREFIX and s.startswith(UPLOAD_BLOCK_PREFIX):
        return True
    if _PDF_EXT_RE.search(s):
        return True
    return False

def _blocked_item(x: dict) -> bool:
    if ALLOW_PDF_CONTEXT:
        return False
    if not isinstance(x, dict):
        return False
    # top-level
    for k in ("url","source","source_url","link","path"):
        v = x.get(k)
        if isinstance(v, str) and _looks_blocked_source(v):
            return True
    # metadata
    md = x.get("metadata") or {}
    if isinstance(md, dict):
        for k in ("url","source","path"):
            v = md.get(k)
            if isinstance(v, str) and _looks_blocked_source(v):
                return True
        if _is_pdf_mime(md.get("mimetype") or md.get("mime") or md.get("content_type")):
            return True
    # payload/data
    pl = x.get("payload") or x.get("data") or {}
    if isinstance(pl, dict):
        for k in ("url","source_url","link","source","path"):
            v = pl.get(k)
            if isinstance(v, str) and _looks_blocked_source(v):
                return True
        if _is_pdf_mime(pl.get("mimetype") or pl.get("mime") or pl.get("content_type")):
            return True
    return False

def _filter_items_for_router(seq):
    if not isinstance(seq, list):
        return seq
    out = []
    for x in seq:
        try:
            if _blocked_item(x):
                continue
        except Exception:
            continue
        out.append(x)
    return out

def _filter_urls(urls: list[str]) -> list[str]:
    return [u for u in (urls or []) if not _looks_blocked_source(u)]

def _extract_confluence_links(html_or_text: str) -> list[dict]:
    out, seen = [], set()
    for m in _A_HREF_RE.finditer(html_or_text or ""):
        url = m.group(1) or ""
        pidm = _PAGEID_RE.search(url)
        if not pidm: continue
        anchor = re.sub(r"<[^>]+>", "", m.group(2) or "").strip()
        key = (pidm.group(1), url, anchor)
        if key in seen: continue
        seen.add(key); out.append({"url": url, "pageId": pidm.group(1), "anchor": anchor})
    for m in _RAW_URL_RE.finditer(html_or_text or ""):
        url = m.group(0); pidm = _PAGEID_RE.search(url)
        if not pidm: continue
        key = (pidm.group(1), url, "")
        if key in seen: continue
        seen.add(key); out.append({"url": url, "pageId": pidm.group(1), "anchor": ""})
    return out

def _apply_acronym_bonus(hits: list[dict], q: str):
    acrs = [a for a in re.findall(r"\b[A-Z]{2,10}\b", q or "")]
    if not acrs: return
    for h in hits:
        md    = (h.get("metadata") or {})
        title = (md.get("title") or "")
        text  = h.get("text")  or ""
        bonus = 0.0
        if any(a in title.upper() for a in acrs): bonus += ACRONYM_TITLE_BONUS
        if any(a in (text[:1200].upper()) for a in acrs): bonus += ACRONYM_BODY_BONUS
        if bonus:
            h["score"] = float(h.get("score") or 0.0) + bonus

def _host_of(u: str) -> str:
    m = _URL_HOST_RE.match(str(u or ""))
    return (m.group(1) or "").lower() if m else ""

def _filter_urls_by_host(urls: list[str]) -> list[str]:
    if not SOURCE_HOST_WHITELIST:
        return urls
    out = []
    for u in urls or []:
        h = _host_of(u)
        if h and any(h == w or h.endswith("." + w) for w in SOURCE_HOST_WHITELIST):
            out.append(u)
    return out

def _filter_mcp_by_strong_tokens(results: list[dict], q: str) -> list[dict]:
    toks  = re.findall(r"[A-Za-z0-9가-힣]{3,}", _to_mcp_keywords(q))
    acrs  = re.findall(r"\b[A-Z]{2,10}\b", q)
    strong = acrs + toks
    if not strong:
        return results
    out = []
    for r in results or []:
        blob = " ".join([(r.get("title") or ""),
                         (r.get("body") or r.get("excerpt") or r.get("text") or "")])
        if any(t in blob for t in strong):
            out.append(r)
    return out or results

def _should_apply_local_bonus(q: str, allowed_spaces: list[str] | None) -> bool:
    """
    로컬 업로드(pool_hits) 가산점 적용 여부를 동적으로 결정.
    - 사용자가 space를 박아줬으면(allowed_spaces) 로컬 보너스 끔
    - 질의가 '도메인 냄새'(_domainish_dynamic) 나면 끔
    - 그 외(일반/모호 질문)엔 켬
    """
    if allowed_spaces:
        return False
    # 내부 도메인 냄새(약어, 토큰-순도 기반)가 나면 로컬 보너스 비활성화
    return not _domainish_dynamic(q or "")

def _should_use_mcp(
    q: str,
    client_spaces: list | None,
    space: str | None,
    reasons: list[str] | None = None,
    local_ok: bool | None = None,
) -> bool:
    """
    MCP 호출 게이트를 '동적 도메인 판별'로 단순화:
    - 명시 space/클라이언트 spaces가 있으면 True
    - 아니면 _domainish_dynamic(q) 결과만 사용
    """
    if space or (client_spaces and len(client_spaces) > 0):
        return True
    if MCP_FALLBACK_ON_EMPTY and local_ok is False:
        return True
    return _domainish_dynamic(q or "")

def _spaces_from_env():
    raw = os.getenv("CONFLUENCE_SPACE", "").strip()
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()] or None

ENV_SPACES = _spaces_from_env()

def _mcp_http_base() -> str:
    if MCP_HTTP_BASE_URL:
        return MCP_HTTP_BASE_URL
    url = os.getenv("MCP_URL", "").strip()
    if not url:
        return ""
    if "/sse" in url:
        url = url.split("/sse", 1)[0]
    return url.rstrip("/")

async def _mcp_page_text_http(page_id: str) -> dict | None:
    if not page_id:
        return None
    base = _mcp_http_base()
    if not base:
        return None
    try:
        async with httpx.AsyncClient(timeout=float(MCP_TIMEOUT)) as client:
            r = await client.get(f"{base}/tool/page_text/{page_id}")
        if r.status_code != 200:
            log.info("MCP page_text http=%s page_id=%s", r.status_code, page_id)
            return None
        j = r.json() if r.content else {}
        text = (j.get("text") or "").strip()
        if not text:
            return None
        return {
            "id": str(j.get("page_id") or page_id),
            "title": (j.get("title") or "").strip(),
            "space": (j.get("space") or "").strip(),
            "url": (j.get("url") or "").strip(),
            "text": text,
        }
    except Exception as e:
        log.warning("MCP page_text failed for page_id=%s: %s", page_id, e)
        return None

def _resolve_allowed_spaces(client_spaces: list | None) -> list | None:
    cs = [s.strip().upper() for s in (client_spaces or []) if isinstance(s, str) and s.strip()]
    if ENV_SPACES and cs:
        inter = [s for s in cs if s in ENV_SPACES]
        return inter or ENV_SPACES
    return ENV_SPACES or (cs or None)

def _mcp_results_to_items(mcp_results: list[dict], k: int) -> tuple[list[dict], list[dict], list[Document]]:
    items, contexts, up_docs = [], [], []
    for r in (mcp_results or [])[:k]:
        title = (r.get("title") or "").strip()
        body  = (r.get("body") or r.get("excerpt") or r.get("text") or "")
        body  = re.sub(r"@@@(?:hl|endhl)@@@", "", body)

        pid = (r.get("id") or "").strip()
        if not pid:
            m_pid = re.search(r"[?&]pageId=(\d+)", (r.get("url") or ""))
            if m_pid: pid = m_pid.group(1)

        raw_url = (r.get("url") or f"confluence:{pid}").strip()
        src_url = _canon_url(raw_url, pid if pid else None)   
        space   = (r.get("space") or "").strip()
        text    = ((title + "\n\n") if title else "") + (body or "")

        md = {
            "source": src_url,     # ← 표준화된 URL을 source/url 모두에
            "url": src_url,
            "kind": "confluence",
            "page": pid or None,
            "pageId": pid or None, 
            "space": space,
            "title": title,
        }

        score = float(r.get("_rank") or r.get("score") or 0.5)  # ← 재랭킹 점수 반영

        items.append({"text": text, "metadata": md, "score": score})
        contexts.append({"text": text, "source": md["source"], "page": md["page"], "kind": md["kind"], "score": score, "title": title,})

        if title:
            up_docs.append(Document(
                page_content=f"[TITLE] {title}",
                metadata={"source": src_url, "title": title, "space": space, "kind": "title", "page": pid or None, "pageId": pid or None}
            ))
        for c in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            up_docs.append(Document(
                page_content=c,
                metadata={"source": src_url, "title": title, "space": space, "kind": "chunk", "page": pid or None, "pageId": pid or None}
            ))
    return items, contexts, up_docs

def _mcp_results_have_text(results: list[dict]) -> bool:
    for r in results or []:
        body = r.get("body") or r.get("excerpt") or r.get("text") or ""
        if str(body).strip():
            return True
    return False



# 빠른 MCP 폴백: 여러 후보 질의를 '동시에' 던지고 '첫 성공'만 사용
# [FIX] 안전한 태스크 취소를 위해 개선
async def _mcp_search_fast(q: str, *, forced_page_id: Optional[str], spaces_for_mcp: list[Optional[str]]) -> list[dict]:
    import asyncio, time
    start = time.monotonic()

    # 1) 질의 후보 구성 (id → 원문 → 키워드화 → 한글 최장토큰)
    cand: list[str] = []
    if forced_page_id:
        cand.append(f"id={forced_page_id}")
    cand.append(q)
    q2 = _to_mcp_keywords(q)
    if q2 and q2 != q:
        cand.append(q2)
    ko = re.findall(r"[가-힣]{2,}", q)
    if ko:
        cand.append(_strip_josa(sorted(ko, key=len, reverse=True)[0]))

    # 중복 제거 + 상한
    seen = set()
    qlist = [x for x in cand if x and not (x in seen or seen.add(x))][:MCP_MAX_TASKS]

    spaces = spaces_for_mcp or [None]

    # [NEW] 안전한 태스크 관리 - TaskGroup 시도, 실패 시 기존 방식
    try:
        # Python 3.11+ TaskGroup 사용 (자동 취소)
        async with asyncio.timeout(MAX_FALLBACK_SECS):
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for sp in spaces:
                    for qq in qlist:
                        tasks.append(tg.create_task(
                            mcp_search(qq, limit=5, timeout=MCP_TIMEOUT, space=sp, langs=SEARCH_LANGS)
                        ))

                # 첫 성공 결과 찾기 (TaskGroup은 모든 태스크 자동 관리)
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await task
                        if result:
                            # TaskGroup은 __aexit__에서 남은 태스크를 자동 취소
                            return result
                    except Exception:
                        continue
    except AttributeError:
        # Python < 3.11 폴백: 기존 방식 (개선된 취소 로직)
        tasks: list[asyncio.Task] = []
        for sp in spaces:
            for qq in qlist:
                tasks.append(asyncio.create_task(
                    mcp_search(qq, limit=5, timeout=MCP_TIMEOUT, space=sp, langs=SEARCH_LANGS)
                ))

        deadline = start + MAX_FALLBACK_SECS
        try:
            while tasks and time.monotonic() < deadline:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.1, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    break

                for t in done:
                    try:
                        part = t.result() or []
                        if part:
                            # 나머지 취소 및 대기
                            for p in pending:
                                p.cancel()
                            # CancelledError 처리 대기
                            await asyncio.gather(*pending, return_exceptions=True)
                            return part
                    except Exception:
                        pass
                tasks = list(pending)
        finally:
            # 타임아웃 시 모든 태스크 안전하게 취소
            for t in tasks:
                if not t.done():
                    t.cancel()
            # CancelledError 전파 대기
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.TimeoutError:
        pass  # TaskGroup 타임아웃

    return []

# PDF 관련 질의 감지 (파일/업로드/확장자/미imetype 기반)
def _is_pdf_related_response(r: dict) -> bool:
    def _meta_is_pdf(md: dict) -> bool:
        mime = (md or {}).get("mimetype") or (md or {}).get("mime") or (md or {}).get("content_type") or ""
        src  = (md or {}).get("source") or (md or {}).get("url") or ""
        return _is_pdf_mime(mime) or bool(_FILE_EXT_RE.search(str(src)))

    # items / contexts 에서 로컬 업로드나 PDF 형태가 섞였는지 확인
    for coll in (r.get("items") or [], r.get("contexts") or []):
        md = (coll.get("metadata") if isinstance(coll, dict) else None) or {}
        if isinstance(coll, dict):  # 단일 hit(dict)인 경우
            src = str(md.get("source") or md.get("url") or "")
            if _is_local_source(src) or _looks_blocked_source(src) or _meta_is_pdf(md):
                return True
        else:  # 리스트인 경우
            for it in coll:
                md = (it.get("metadata") or {})
                src = str(md.get("source") or md.get("url") or "")
                if _is_local_source(src) or _looks_blocked_source(src) or _meta_is_pdf(md):
                    return True

    # source_urls 에 PDF/업로드 흔적이 있으면 역시 PDF 관련으로 간주
    for u in (r.get("source_urls") or []):
        if _looks_blocked_source(u) or _FILE_EXT_RE.search(str(u)):
            return True

    return False

def _anchor_tokens_from_query(q: str) -> list[str]:
    # _tokenize_query는 파일에 이미 정의되어 있음
    toks = _tokenize_query(q)
    return [t for t in toks if len(t) >= 3 and t not in _GENERIC]

def _collect_text_of_source(src: str, limit_chars: int = 2000) -> str:
    """docstore에서 해당 source 텍스트/제목을 조금 모아 샘플링"""
    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
    except Exception:
        dct = {}
    norm = _norm_source(src)
    buf_title, buf_body = [], []
    for d in dct.values():
        md = d.metadata or {}
        if _norm_source(str(md.get("source",""))) != norm:
            continue
        if md.get("title"):
            buf_title.append(str(md.get("title")))
        if d.page_content:
            buf_body.append(d.page_content)
        if sum(len(x) for x in buf_body) > limit_chars:
            break
    title_part = " ".join(buf_title)
    body_part  = " ".join(buf_body)[:limit_chars]
    return (title_part + "\n" + body_part).strip()

# [NEW] 제네릭 질문 패턴 (sticky 해제용)
_GENERIC_QUESTION_RE = re.compile(
    r'^(안녕|고마워|감사|좋아|최고|괜찮|오케이|ok|okay|yes|no|응|아니|'
    r'알았어|됐어|됐다|그래|네|예|아니오|아뇨|ㅇㅋ|ㅇㅇ|ㄴㄴ|'
    r'날씨|시간|요일|몇\s*시|뭐\s*해|심심|재밌|재미|놀|게임).*',
    re.I
)

def _sticky_is_relevant(q: str, src: str) -> bool:
    """Sticky 소스가 현재 질문과 관련있는지 검사"""
    # [NEW] 제네릭 질문은 sticky 해제
    if _GENERIC_QUESTION_RE.match(q.strip()):
        return False

    anchors = _anchor_tokens_from_query(q)
    if not anchors:
        # [FIX] 앵커가 없어도 제네릭 아니면 일단 유지 (짧은 질문 허용)
        # 하지만 너무 짧은 질문(<3자)은 거부
        if len(q.strip()) < 3:
            return False
        return True

    sample = _collect_text_of_source(src, limit_chars=3000).lower()
    if not sample:
        return False
    hit = sum(1 for a in anchors if a.lower() in sample)
    need = 1 if len(anchors) <= 2 else max(2, int(len(anchors)*0.5))
    return hit >= need

def _is_datetime_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    # RAG 신호가 있으면 직접답 차단
    if _EXPLICIT_DATE_RE.search(q) or _RETRIEVAL_HINT_RE.search(q):
        return False
    # '오늘/지금/현재' 같은 지시어가 반드시 있어야 함
    if not _DEICTIC_RE.search(q):
        return False
    # 도메인 냄새가 강하면(사내 문서 맥락) 즉답 대신 RAG/MCP로 보내기
    if _domainish_dynamic(q):
        return False
    # 실제로 날짜/시간 질의인지
    return bool(_DATE_TIME_NEED_RE.search(q))


def _collect_source_urls_from_contexts(ctxs: list[dict], top_n: int = 16) -> list[str]:
    rows, seen, out = [], set(), []
    for c in ctxs or []:
        u   = c.get("source") or ""
        pid = str(c.get("page") or "")
        cu  = _canon_url(u, pid if pid else None)
        if not cu: continue
        title = c.get("title") or ""
        score = float(c.get("score") or 0.0)
        is_conf = (("pageId=" in cu) or (c.get("kind") == "confluence"))
        is_bad  = bool(_BAD_TITLE_RE.search(title))
        rows.append((cu, score, is_conf, is_bad))

    # Confluence 우선, 회의록/스크럼 후순위, 점수 내림차순
    rows.sort(key=lambda r: (r[2], not r[3], r[1]), reverse=True)

    for cu, *_ in rows:
        if cu not in seen:
            seen.add(cu); out.append(cu)
            if len(out) >= top_n: break
    return out


def _now_str_kst() -> tuple[str, str, str]:
    try:
        now = datetime.now(ZoneInfo(TZ_NAME))
    except Exception:
        # tzdata 미설치 등 문제 시 로컬 타임 폴백
        now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    dow_str  = ["월","화","수","목","금","토","일"][now.weekday()]
    time_str = now.strftime("%H:%M")
    return date_str, dow_str, time_str

async def _call_llm(messages: list[dict], **kwargs) -> str:
    """call_chat_completions이 sync/async 어느 쪽이든 안전하게 호출"""
    try:
        if inspect.iscoroutinefunction(call_chat_completions):
            return await call_chat_completions(messages=messages, **kwargs)
        return call_chat_completions(messages=messages, **kwargs)
    except TypeError:
        # 혹시 위치인자만 받는 구현일 수도 있어서 백업
        if inspect.iscoroutinefunction(call_chat_completions):
            return await call_chat_completions(messages)
        return call_chat_completions(messages)

def _extract_page_id(src: Optional[str]) -> Optional[str]:
    """Confluence URL 에서 pageId= 숫자만 뽑는다."""
    if not src:
        return None
    m = _PAGEID_RE.search(str(src))
    return m.group(1) if m else None

def _has_confl_hint(q: str) -> bool:
    return any(w.lower() in q.lower() for w in CONFL_HINT_WORDS)


def _url_has_page_id(url: Optional[str], page_id: Optional[str]) -> bool:
    """url 이 해당 pageId 를 가리키는지 여부"""
    if not url or not page_id:
        return False
    return bool(_PAGEID_RE.search(str(url)) and page_id in str(url))

def _set_sticky(src: str, secs: int = STICKY_SECS, session_id: Optional[str] = None):
    """해당 소스를 잠시 기본 대상으로 고정(연속 질문 시 섞임 방지)"""
    global current_source, current_source_until
    # 전역 sticky (후방 호환)
    current_source = _norm_source(src)
    current_source_until = time.time() + secs
    # 세션 기반 sticky (비동기 실행)
    if session_id:
        try:
            asyncio.create_task(session_manager.set_sticky(session_id, src, secs))
        except Exception:
            pass  # 이벤트 루프 없으면 무시

def _norm_source(s: str) -> str:
    s = (s or "").replace("\\", "/")
    return re.sub(r"^/app/", "", s)  # 컨테이너 절대경로를 제거해 'uploads/…'로 맞춤

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _norm_kr(s: str) -> str:
    # [DEPRECATED] 하위 호환용 - normalize_text_for_search 사용 권장
    # 공백 제거 + 도메인 동의어/정규화 적용 + 소문자
    t = _basic_normalize(s)
    t = _apply_canon_map(t)
    t = re.sub(r"\s+", "", t)        # 공백 제거 (아파트 누리 -> 아파트누리)
    return t.lower()

# === [NEW] 통일된 정규화 함수 (검색/인덱싱 공통) ===
def normalize_text_for_search(text: str, preserve_spaces: bool = False) -> str:
    """
    검색과 인덱싱에 모두 사용되는 정규화 함수

    Args:
        text: 정규화할 텍스트
        preserve_spaces: True면 공백 유지, False면 한글 연속부 공백 제거

    Returns:
        정규화된 텍스트
    """
    if not text:
        return ""

    # 1단계: 기본 정규화 (NFC, 공백 축약)
    t = _basic_normalize(text)

    # 2단계: 도메인 동의어 치환
    t = _apply_canon_map(t)

    # 3단계: 한국어 복합명사 공백 처리
    if not preserve_spaces:
        t = _collapse_korean_compounds(t)

    # 4단계: 소문자 변환
    t = t.lower()

    return t

def _match_pageid(md: dict, pid: str) -> bool:
    if not pid:
        return True
    pid_md = str((md or {}).get("page") or (md or {}).get("pageId") or "")
    src = str((md or {}).get("source") or "")
    url = str((md or {}).get("url") or "")
    return (pid_md == pid) or (f"pageId={pid}" in src) or (f"pageId={pid}" in url)

### [ADD] 로컬 업로드 소스 판별(경로 정규화 포함)
def _is_local_source(src: str) -> bool:
    s = _norm_source((src or "").replace("\\", "/"))
    up = (UPLOAD_DIR or "uploads").replace("\\", "/").strip("/")
    return (
        s.startswith("uploads/") or
        s.startswith(f"{up}/") or
        f"/{up}/" in f"/{s}" or
        (s.endswith(".pdf") and f"/{up}/" in f"/{s}")  # 옵션
    )

def _apply_local_bonus(hits: list[dict]):
    if not LOCAL_FIRST:
        return
    for h in hits:
        md = (h.get("metadata") or {})
        src = str(md.get("source") or md.get("url") or "")
        ### [FIX] 경로 표준화 기반의 로컬 판별 사용
        if _is_local_source(src):
            h["score"] = float(h.get("score") or 0.0) + LOCAL_BONUS

def _apply_sticky_bonus(hits: list[dict], sticky_src: str | None):
    """sticky를 강제 필터 대신 '가산점'으로만 반영"""
    if not sticky_src:
        return
    want = _norm_source(sticky_src)
    for h in hits or []:
        md = (h.get("metadata") or {})
        src = _norm_source(str(md.get("source") or md.get("url") or ""))
        if src == want:
            h["score"] = float(h.get("score") or 0.0) + STICKY_BONUS

def _is_title_like(q: str) -> bool:
    # 긴 한글 구절이 있고, 전체 토큰이 과하지 않으면 '제목형'으로 판단
    return bool(_longest_ko_phrase(q)) and (len(_tokenize_query(q)) <= 4)

def _rebalance_by_source(hits: list[dict], k: int, per_source_cap: int) -> list[dict]:
    """소스 다양성 보장: 소스당 최대 per_source_cap개, 라운드로빈으로 k개 선택"""
    if per_source_cap <= 0:
        return hits[:k]
    # 점수순 그룹화
    groups: dict[str, list[dict]] = {}
    for h in sorted(hits, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        key = _norm_source(str((h.get("metadata") or {}).get("source") or ""))
        groups.setdefault(key, []).append(h)

    out: list[dict] = []
    # 1차: 라운드로빈
    while len(out) < k and any(groups.values()):
        for key in list(groups.keys()):
            lst = groups.get(key) or []
            if not lst:
                groups.pop(key, None); continue
            used = sum(1 for x in out
                       if _norm_source(str((x.get("metadata") or {}).get("source") or "")) == key)
            if used < per_source_cap:
                out.append(lst.pop(0))
                if len(out) >= k:
                    break
            else:
                groups[key] = []  # 이 소스는 상한 도달

    # 2차: 아직 모자라면 남은 후보로 채우되 상한을 넘기지 않음
    if len(out) < k:
        leftovers: list[dict] = []
        for lst in groups.values():
            leftovers.extend(lst)
        leftovers.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        for h in leftovers:
            key = _norm_source(str((h.get("metadata") or {}).get("source") or ""))
            used = sum(1 for x in out
                       if _norm_source(str((x.get("metadata") or {}).get("source") or "")) == key)
            if used < per_source_cap:
                out.append(h)
                if len(out) >= k:
                    break
    return out[:k]

def _apply_page_hint(hits: List[dict], page_id: str | None):
    """pageId가 오면 '가산점'만 (soft). 잠그지 않음."""
    if not page_id or PAGE_FILTER_MODE.lower() != "soft":
        return
    for h in hits:
        if _match_pageid(h.get("metadata") or {}, page_id):
            h["score"] = float(h.get("score") or 0.0) + PAGE_HINT_BONUS

def _coalesce_single_source(hits: List[dict], q: str) -> List[dict]:
    """
    '최근/이슈/목록/정리' 같은 질의면 여러 출처가 섞이지 않게
    가장 점수가 높은 '단일 소스(=한 페이지)'만 남긴다.
    """
    if not SINGLE_SOURCE_COALESCE or not hits or not _COALESCE_TRIGGER.search(q or ""):
        return hits

    # source+page를 키로 클러스터링
    groups = {}
    for h in hits:
        md = h.get("metadata") or {}
        key = f"{md.get('url') or md.get('source') or ''}|p{md.get('page') or md.get('pageId') or ''}"
        groups.setdefault(key, []).append(h)

    scored = []
    for key, hs in groups.items():
        s = sum(float(x.get("score") or 0.0) for x in hs)
        scored.append((key, s, hs))
    if not scored:
        return hits

    scored.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score, best_group = scored[0]
    total = sum(s for _, s, _ in scored) or 1.0
    if (best_score / total) >= COALESCE_THRESHOLD or len(scored) > 1:
        # 한 페이지로 수렴
        try:
            # _set_sticky(best_key.split("|p", 1)[0])  # 다음 질문 안정화
            if STICKY_FROM_COALESCE:
                try:
                    _set_sticky(best_key.split("|p", 1)[0])
                except Exception:
                    pass
        except Exception:
            pass
        return best_group
    return hits

SEARCH_LANGS = [s.strip() for s in os.getenv("SEARCH_LANGS", "ko,en").split(",") if s.strip()]

# 키워드 정제(Confluence CQL용)
_K_STOP = {"관련","내용","찾아줘","찾아","알려줘","정리","컨플루언스","에서","해줘",
           "무엇","어떤","대한","관련한","좀","좀만","계속","그리고","거나",
           "설명","소개","정의","토큰","token","코인", "해주세요","해 주세요","해줘요","해줘","주세요","대하여","대해","설명해줘","설명해주세요","말해줘","알려","알려줘","부탁해"
           }

_JOSA_RE = re.compile(
    r"(으로써|으로서|으로부터|라고는|라고도|라고|처럼|까지|부터|에게서|한테서|에게|한테|께|이며|이자|"
    r"으로|로서|로써|로부터|께서|와는|과는|에서는|에는|에서|에게|한테|와|과|을|를|은|는|이|가|의|에|도|만|랑|하고)$"
)

ACRONYM_RE = re.compile(r'^[A-Za-z]{2,5}$')

def _split_core_and_acronyms(tokens: List[str]) -> Tuple[List[str], List[str]]:
    core, acr = [], []
    for t in tokens:
        if ACRONYM_RE.match(t):
            acr.append(t.upper())     # ← 약어는 모두 대문자로 통일
        else:
            core.append(t)
    return core, acr

_TOK_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")

# 불용구/공손표현 꼬리 자르기 + 분절
_STOP_SUFFIX_RE = re.compile(
    r"(에\s*대하여|에\s*대해|에\s*대한|대하여|대해|"
    r"설명해\s*주세요|설명해주세요|설명해줘|알려줘|해줘요?|주세요)$"
)

def _preseg_stop_phrases(q: str) -> str:
    # 붙여쓰기일 때도 뭉친 꼬리를 띄우거나 제거
    return _STOP_SUFFIX_RE.sub(" ", q)

def _tokenize_query(q: str) -> List[str]:
    q = _preseg_stop_phrases(_basic_normalize(q))
    raw = _TOK_RE.findall(q or "")
    toks = []
    for t in raw:
        t = _STOP_SUFFIX_RE.sub("", t)  # 토큰 끝 꼬리 날리기
        t = _strip_josa(t)
        if t and t not in _K_STOP:
            toks.append(t)
    return list(dict.fromkeys(toks))[:12]

def _keyword_overlap_score(q_tokens: List[str], text: str, title: str = "") -> float:
    """간단한 스파스 점수: 토큰 교집합 비율 + 제목 매치 보너스"""
    if not q_tokens:
        return 0.0
    # 텍스트는 길 수 있으니 앞부분만 얇게 보지만, 제목은 풀로 본다
    body = (text or "")[:1200]
    hits = sum(1 for t in q_tokens if t in body)
    base = hits / max(4, len(q_tokens))          # 토큰 일부만 맞아도 0.x 점수
    if title:
        title_hits = sum(1 for t in q_tokens if t in title)
        if title_hits:
            base += TITLE_BONUS         # 제목 매치 보너스 (가산점)
    return float(base)

def _sparse_keyword_hits(q: str, limit: int = 150, space: Optional[str] = None) -> List[dict]:
    """
    docstore 전체를 훑어 '간단 키워드 매치' 기반 후보를 만든다.
    반환: pool_hits와 동일한 dict 목록({text, metadata, score})

    [FIX] max_scan 제한 제거, 최신 문서 우선 스캔
    """
    if not q.strip():
        return []
    q_tokens = _tokenize_query(q)
    if not q_tokens:
        return []

    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
    except Exception:
        return []

    # [NEW] 문서들을 최신순으로 정렬 (created_at이 있으면 사용, 없으면 소스명 역순)
    docs_sorted = sorted(
        dct.values(),
        key=lambda d: (
            -(d.metadata or {}).get("created_at", 0),  # 최신순
            (d.metadata or {}).get("source", "")       # 소스명
        )
    )

    out = []
    for d in docs_sorted:
        md = dict(d.metadata or {})
        src = str(md.get("source", ""))

        # limit 도달 시 중단 (max_scan 제한 제거)
        if len(out) >= limit:
            break
        
        # (옵션) space 하드필터
        if space and SPACE_FILTER_MODE.lower() == "hard":
            s = (md.get("space") or "").strip()
            if not s or s.lower() != space.lower():
                continue

        title = (md.get("title") or "")
        score = _keyword_overlap_score(q_tokens, d.page_content or "", title)
        if score <= 0.0:
            continue

        # pool_hits 포맷으로 변환
        md["source"] = src
        out.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": min(0.99, score),  # 스파스 자체 점수는 0~1 사이
        })

    # 상위 limit만
    out.sort(key=lambda h: h["score"], reverse=True)
    return out[:int(limit)]

# ===== 제목/한글구절 정규화 유틸 =====
def _norm_ko_text(s: str) -> str:
    """제목/한글구절 정규화 - normalize_text_for_search 래퍼"""
    return normalize_text_for_search(s, preserve_spaces=False)

def _longest_ko_phrase(q: str) -> str:
    q = _preseg_stop_phrases(_basic_normalize(q)) 
    cands = re.findall(r"[가-힣]{2,}(?:\s+[가-힣]{2,})*", q or "")
    if not cands:
        return ""
    # 조사도 한 번 더 걷어내고 정규화
    norm = [_norm_ko_text(_strip_josa(c)) for c in cands]
    norm = [c for c in norm if c]
    return max(norm, key=len) if norm else ""

def _dedup_by_title(results: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in results or []:
        t = _norm_ko_text(r.get("title") or "")
        if t and t not in seen:
            seen.add(t); out.append(r)
        elif not t:
            out.append(r)
    return out

def _rerank_mcp_results(q: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    keyphrase = _longest_ko_phrase(q)       # 이제 ‘상가정보’
    q_tokens  = _tokenize_query(q)          # ['NIA','상가정보'] 등
    acrs      = [a.upper() for a in re.findall(r"\b[A-Za-z]{2,10}\b", q)]

    # 추가: 약어+키프레이즈 결합(예: 'NIA상가정보')
    combo = _norm_ko_text((acrs[0] + keyphrase) if (acrs and keyphrase) else "")

    scored = []
    for r in results:
        title = r.get("title") or ""
        body  = r.get("body") or r.get("excerpt") or r.get("text") or ""
        ntitle = _norm_ko_text(title)
        title_u = title.upper()
        s = float(r.get("score") or 0.0)
        try:
            ver = int(r.get("version") or 0)
        except Exception:
            ver = 0
        if ver > 0:
            s += min(0.8, (math.log10(ver + 1) / 2.0))

        # ① 제목 정확/부분 일치
        if keyphrase and ntitle:
            if ntitle == keyphrase:      s += 2.0
            elif keyphrase in ntitle:    s += 1.0

        # ② 약어+키프레이즈 결합 타이틀 매치(강한 보정)
        if combo and ntitle:
            if ntitle == combo:          s += 2.2
            elif combo in ntitle:        s += 1.4

        # ③ 토큰 커버리지
        if q_tokens:
            tl = title.lower()
            s += 0.12 * sum(1 for t in q_tokens if t.lower() in tl)

        # ④ 약어가 제목에 있으면 약간 보너스
        if acrs and any(a in title_u for a in acrs):
            s += 0.2

        if LOGIN_PAT.search(body or ""):
            s -= 0.5
        r["_rank"] = s
        scored.append(r)

    # 집계 페이지 본문에 있는 하이퍼링크의 앵커 텍스트가 질문의 핵심 구절(keyphrase)과 '정확히' 일치하면 타겟 pageId 점수 크게 올리기
    # [FIX] 90.0 → 1.5로 정규화 (다른 보너스와 밸런스 맞춤)
    ANCHOR_LINK_BOOST = 1.5  # 앵커 링크 매칭 보너스
    ANCHOR_LINK_PENALTY = 0.1  # 중간 페이지 페널티

    def _pid_of(res: dict) -> str | None:
        pid = (res.get("id") or "") or ""
        if not pid:
            m = _PAGEID_RE.search(res.get("url") or "")
            pid = m.group(1) if m else ""
        return pid or None

    by_pid = {}
    for res in scored:
        pid = _pid_of(res)
        if pid: by_pid[pid] = res

    exact_title_norm = _norm_ko_text(keyphrase or "")
    targets = [exact_title_norm]
    if combo:
        targets.append(_norm_ko_text(combo))  # NIA+제목 형태도 허용

    if exact_title_norm:
        for res in scored:
            body = res.get("body") or res.get("excerpt") or res.get("text") or ""
            for ln in _extract_confluence_links(body):
                anchor_norm = _norm_ko_text(ln.get("anchor") or "")
                if anchor_norm and any(t and (anchor_norm == t or t in anchor_norm) for t in targets):
                    tgt = by_pid.get(ln["pageId"])
                    if tgt and tgt is not res:
                        tgt["_rank"] = float(tgt.get("_rank") or 0.0) + ANCHOR_LINK_BOOST
                        res["_rank"] = float(res.get("_rank") or 0.0) - ANCHOR_LINK_PENALTY

    scored.sort(key=lambda x: x.get("_rank", 0.0), reverse=True)
    return _dedup_by_title(scored)


# 1) 기본 정규화: 호환성 정규화 + 공백 축약
def _basic_normalize(text: str) -> str:
    # NFC로 정규화 (한글 결합형 안정화)
    t = unicodedata.normalize("NFC", text)
    # 전각/반각, 제어문자 정리 (필요하면 NFKC 고려)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 2) 한국어 특화: 의미보존 범위에서 공백 제거(명사연속부)
#    - "지역 정보" → "지역정보" 같은 변이를 줄이기 위해 한글+숫자 연속부 내 공백 제거
_HANGUL_BLOCK = r"[가-힣0-9A-Za-z]"
def _collapse_korean_compounds(text: str) -> str:
    # 한글/숫자/영문 사이의 단일 공백 제거
    return re.sub(fr"(?<={_HANGUL_BLOCK})\s+(?={_HANGUL_BLOCK})", "", text)

# ① 환경변수 주입식(운영에서 교체 가능)
CANON_MAP = json.loads(os.getenv("CANON_MAP_JSON", "{}") or "{}")

# ② 없으면 동적 유도(타이틀 n-gram 빈도 기반)
def derive_canon_map_from_index(top_k: int = 50) -> dict[str, str]:
    # 보강: vectorstore가 아직 없을 때 안전하게 빈 맵 반환
    try:
        ds = getattr(vectorstore, "docstore", None)
    except NameError:
        return {}
    dct = getattr(ds, "_dict", {}) if ds else {}
    freq = Counter()
    for d in dct.values():
        t = str((d.metadata or {}).get("title",""))
        m = re.findall(r"[가-힣]{2,}\s+[가-힣]{2,}", t)
        for phrase in m:
            freq[phrase] += 1
    cmap = {}
    for p, _ in freq.most_common(top_k):
        cmap[p] = re.sub(r"\s+", "", p)
    return cmap

# 여기서 즉시 유도하지 말고 비워둠 (startup에서 채워짐)
if not CANON_MAP:
    CANON_MAP = {}

def _apply_canon_map(text: str) -> str:
    t = text
    for pat, rep in CANON_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def normalize_query(q: str) -> str:
    """쿼리 정규화 - normalize_text_for_search 래퍼"""
    return normalize_text_for_search(q, preserve_spaces=False)

def _apply_space_hint(hits: List[dict], space: Optional[str]):
    """SPACE_FILTER_MODE == soft 일 때, space 일치 항목에 가산점"""
    if not space or SPACE_FILTER_MODE.lower() != "soft":
        return
    sp = space.lower()
    for h in hits:
        s = ((h.get("metadata") or {}).get("space") or "").lower()
        if s and s == sp:
            h["score"] = float(h.get("score") or 0.0) + SPACE_HINT_BONUS

def _strip_josa(t: str) -> str:
    return _JOSA_RE.sub("", t)

def _to_mcp_keywords(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9가-힣]+", q)
    toks = [_strip_josa(t) for t in toks]                # ← 조사 제거
    toks = [t for t in toks if t and t not in _K_STOP and len(t) >= 2]
    hangs = [t for t in toks if re.search(r"[가-힣]", t)]
    keep = hangs if hangs else toks
    # 중복 제거 + 과도한 길이 방지
    dedup = list(dict.fromkeys(keep))
    return " ".join(dedup[:6])[:200] or q

logger = logging.getLogger("rag-proxy")
log = logging.getLogger(__name__)

_reranker = None
_reranker_init_error = None

def _get_reranker():
    global _reranker, _reranker_init_error
    if not bool(getattr(settings, "ENABLE_RERANKER", False)):
        return None
    if _reranker is None and _reranker_init_error is None:
        try:
            _reranker = FlagReranker(settings.RERANKER_MODEL, use_fp16=True)
        except Exception as exc:
            _reranker_init_error = str(exc)
            log.warning("reranker init failed: %s", exc)
            _reranker = None
    return _reranker

def _rerank_pool_hits(q: str, pool_hits: list[dict]) -> list[dict]:
    if not pool_hits:
        return pool_hits
    reranker = _get_reranker()
    if not reranker:
        return pool_hits

    cand_n = int(getattr(settings, "RERANKER_CANDIDATES", 60))
    top_n = int(getattr(settings, "RERANKER_TOP_N", 5))
    if cand_n <= 0 or top_n <= 0:
        return pool_hits

    ranked = sorted(pool_hits, key=lambda h: float(h.get("score") or 0.0), reverse=True)
    candidates = ranked[: max(cand_n, top_n)]
    if not candidates:
        return pool_hits

    pairs = []
    for h in candidates:
        md = h.get("metadata") or {}
        text = h.get("text") or md.get("title") or ""
        pairs.append((q, text))

    try:
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        log.warning("reranker compute failed: %s", exc)
        return pool_hits

    for h, score in zip(candidates, scores):
        h["rerank_score"] = float(score)

    candidates.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
    cand_ids = {id(h) for h in candidates}
    rest = [h for h in ranked if id(h) not in cand_ids]
    return candidates + rest

# 전역 상태
retriever = None
vectorstore: Optional[FAISSStore] = None
drop_think_fn = None

INDEX_DIR = settings.INDEX_DIR
UPLOAD_DIR = settings.UPLOAD_DIR
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DEVICE = settings.EMBEDDING_DEVICE
DATA_CSV = settings.DATA_CSV

def ensure_dirs():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

class _PrefixedEmbeddings(Embeddings):
    def __init__(self, base: Embeddings, use_prefix: bool):
        self._base = base
        self._use_prefix = use_prefix

    def _embed_query(self, text: str) -> List[float]:
        fn = getattr(self._base, "embed_query", None)
        if callable(fn):
            return fn(text)
        fn = getattr(self._base, "embed_documents", None)
        if callable(fn):
            return fn([text])[0]
        fn = getattr(self._base, "embed_texts", None)
        if callable(fn):
            return fn([text])[0]
        raise RuntimeError("embed_query not supported by base embeddings")

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        fn = getattr(self._base, "embed_documents", None)
        if callable(fn):
            return fn(texts)
        fn = getattr(self._base, "embed_query", None)
        if callable(fn):
            return [fn(t) for t in texts]
        fn = getattr(self._base, "embed_texts", None)
        if callable(fn):
            return fn(texts)
        raise RuntimeError("embed_documents not supported by base embeddings")

    def embed_query(self, text: str) -> List[float]:
        if self._use_prefix:
            text = f"query: {text}"
        return self._embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._use_prefix:
            texts = [f"passage: {t}" for t in texts]
        return self._embed_documents(texts)


class _SafeEmbeddings(Embeddings):
    def __init__(self, base: Embeddings):
        self._base = base

    def embed_query(self, text: str) -> List[float]:
        fn = getattr(self._base, "embed_query", None)
        if callable(fn):
            return fn(text)
        fn = getattr(self._base, "embed_documents", None)
        if callable(fn):
            return fn([text])[0]
        fn = getattr(self._base, "embed_texts", None)
        if callable(fn):
            return fn([text])[0]
        raise RuntimeError("embed_query not supported by embeddings")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        fn = getattr(self._base, "embed_documents", None)
        if callable(fn):
            return fn(texts)
        fn = getattr(self._base, "embed_query", None)
        if callable(fn):
            return [fn(t) for t in texts]
        fn = getattr(self._base, "embed_texts", None)
        if callable(fn):
            return fn(texts)
        raise RuntimeError("embed_documents not supported by embeddings")


def _make_embedder() -> Embeddings:
    emb: Embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    if getattr(settings, "E5_USE_PREFIX", False) and "e5" in EMBEDDING_MODEL.lower():
        emb = _PrefixedEmbeddings(emb, use_prefix=True)
    return _SafeEmbeddings(emb)

def _load_or_init_vectorstore() -> FAISSStore:
    """
    FAISS 인덱스 로드 또는 초기화

    [FIX] 인덱스 파일이 있는데 로드 실패하면 명시적 에러 발생
    """
    faiss_file = Path(INDEX_DIR) / "index.faiss"
    pkl_file   = Path(INDEX_DIR) / "index.pkl"

    # 인덱스 파일이 존재하는 경우
    if faiss_file.exists() and pkl_file.exists():
        try:
            emb = _make_embedder()
            vs = FAISSStore.load_local(
                INDEX_DIR,
                embeddings=emb,
                allow_dangerous_deserialization=True
            )
            log.info("✓ Loaded existing FAISS index from %s (docs=%d)",
                     INDEX_DIR, len(vs.docstore._dict))
            return vs
        except Exception as e:
            # [NEW] 인덱스 파일이 있는데 로드 실패 → 치명적 에러
            log.error("✗ CRITICAL: Failed to load existing FAISS index: %s", e)
            log.error("  Index files exist but are corrupted. Please:")
            log.error("  1. Check index integrity: %s", INDEX_DIR)
            log.error("  2. Restore from backup, or")
            log.error("  3. Delete index files to rebuild")
            raise RuntimeError(
                f"Index corrupted: {e}. "
                "Please restore from backup or delete index directory to rebuild."
            ) from e

    # 인덱스 파일이 없는 경우 → 빈 인덱스로 시작
    log.info("○ No existing index found, initializing empty FAISS index")
    vs = _empty_faiss()
    return vs

def _empty_faiss() -> FAISSStore:
    emb = _make_embedder()
    dim = len(emb.embed_query("dim-probe"))
    index = faiss.IndexFlatL2(dim)   # normalize_embeddings=True이지만 기본 L2로 통일
    return FAISSStore(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

def _reload_retriever():
    global retriever
    retriever = vectorstore.as_retriever(
        search_type=settings.SEARCH_TYPE,                       
        search_kwargs={"k": settings.RETRIEVER_K,               
                       "fetch_k": settings.RETRIEVER_FETCH_K}   
    )
    VS.retriever = retriever


# 인덱싱 동시성 방지 락
index_lock = asyncio.Lock()

# 문서 고유 ID (중복 방지용)
def _doc_id(d: Document) -> str:
    src   = str(d.metadata.get("source", ""))
    page  = str(d.metadata.get("page", ""))
    kind  = str(d.metadata.get("kind", ""))
    chunk = str(d.metadata.get("chunk", ""))
    # 내용 기반 해시를 섞어 동일 page/kind라도 서로 다르게
    h = hashlib.sha1((d.page_content or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{src}|p{page}|k{kind}|c{chunk}|h{h}"

def _upsert_docs_no_dup(docs: List[Document]) -> int:
    global vectorstore
    if not docs:
        return 0
    exist = set(getattr(vectorstore.docstore, "_dict", {}).keys())
    batch = set()  # ← 한 번의 add_documents 안에서의 중복을 걸러냄
    new_docs, ids = [], []
    for d in docs:
        did = _doc_id(d)
        if did in exist or did in batch:
            continue
        new_docs.append(d)
        ids.append(did)
        batch.add(did)
    if not new_docs:
        return 0
    vectorstore.add_documents(new_docs, ids=ids)
    vectorstore.save_local(INDEX_DIR)
    _reload_retriever()
    return len(new_docs)

def build_openai_messages(question: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[Document]]:
    global retriever, vectorstore

    # 1) 메타 질문이면 k 확장
    k_eff = 12 if META_PAT.search(question) else k

    # 2) 리트리브
    docs: List[Document] = []
    if retriever is not None:
        try:
            docs = retriever.invoke(question)
        except Exception:
            docs = []

    # 3) 질문에 파일명 힌트가 있으면 같은 파일 청크 우선
    m = re.search(r'([^\s"\'()]+\.(?:pdf|pptx|ppt|xlsx|xls|csv|txt|md))', question, re.I)
    fname = m.group(1).lower() if m else None
    if fname and docs:
        filt = [d for d in docs if fname in str(d.metadata.get("source","")).lower()]
        if filt:
            docs = filt

    # 4) 제목/요약 청크를 docstore에서 찾아 맨 앞에 프리패스(있을 때)
    if fname and vectorstore is not None:
        try:
            ds = vectorstore.docstore._dict.values()
            title_doc = next((d for d in ds
                              if fname in str(d.metadata.get("source","")).lower()
                              and d.metadata.get("kind") == "title"), None)
            summary_doc = next((d for d in ds
                                if fname in str(d.metadata.get("source","")).lower()
                                and d.metadata.get("kind") == "summary"), None)
            prepend = []
            if title_doc and title_doc not in docs:
                prepend.append(title_doc)
            if summary_doc and summary_doc not in docs:
                prepend.append(summary_doc)
            if prepend:
                docs = prepend + docs
        except Exception:
            pass

    # 5) 제목/요약이 앞에 오도록 정렬(리트리브로 들어온 경우까지 정리)
    if docs:
        docs = sorted(docs, key=lambda d: 0 if d.metadata.get("kind") in ("title","summary") else 1)

    # 6) 컨텍스트 합치기
    context = "\n\n---\n\n".join(d.page_content for d in docs[:k_eff]) if docs else ""

    # 7) 시스템 프롬프트
    system = lang_wrap(
        "너의 사고과정은 절대로 출력하지 말고, 다음 컨텍스트로만 간결하고 정확하게 한국어로 답하라. "
        "모르면 정확히 다음 문장을 출력하라: 주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다"
    )
    if META_PAT.search(question):
        system += "\n\n요청이 '주제/개요/요약' 성격이면 핵심 내용을 2-3문장으로 요약하라."
    if context:
        system += f"\n\n컨텍스트:\n{context}"
    
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return msgs, docs[:k_eff]

@app.on_event("startup")
def _startup():
    """데이터 CSV가 있으면 그걸로 인덱스 시작, 없으면 '빈 인덱스'로 시작(=PDF 바로 테스트 가능)."""
    global vectorstore, drop_think_fn
    ensure_dirs()
    try:
        # CSV가 있으면 기존 파이프라인으로 로딩
        if DATA_CSV and isinstance(DATA_CSV, str) and Path(DATA_CSV).exists():
            _rag_chain, _retr, vectorstore, drop_think_fn = build_rag_chain(
                data_path=DATA_CSV, index_dir=INDEX_DIR
            )
        else:
            vectorstore = _load_or_init_vectorstore()
        _reload_retriever()
        logger.info("Startup complete. Index at %s", INDEX_DIR)
    except Exception as e:
        logger.exception("Startup failed: %s", e)
    VS.vectorstore = vectorstore
    VS.retriever = retriever
    rebuild_domain_stats_from_index()


@app.get("/health")
def health():
    try:
        n = len(vectorstore.docstore._dict) if vectorstore else None
    except Exception:
        n = None
    return {"status": "ok" if vectorstore else "degraded",
            "doc_count": n, "index_dir": INDEX_DIR, "uploads": UPLOAD_DIR}

@app.get("/models")
async def models():
    return await proxy_get("/models")

@app.get("/v1/models")
def v1_models():
    import time
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-30b-a3b-fp8-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "router"
            }
        ]
    }


from pydantic import Field

class AskPayload(BaseModel):
    """
    OpenAPI에 'q'가 반드시 필요한 필드로 표기되도록 하는 요청 스키마.
    - Open WebUI가 이 스키마를 읽고 q를 포함해 호출하게 됨
    - top_k / k, space / spaceKey 등도 받아서 query()에 그대로 전달
    """
    q: str = Field(..., description="사용자 질문 (필수)")
    k: int | None = Field(5, description="리트리브 상위 K", ge=1, le=50)
    top_k: int | None = Field(None, description="일부 클라이언트 호환용(내부에서 k로 매핑)")
    source: str | None = Field(None, description="소스 단일 필터")
    sources: list[str] | None = Field(None, description="소스 다중 필터")
    space: str | None = Field(None, description="Confluence space 힌트")
    spaceKey: str | None = Field(None, description="space와 동일(클라이언트 호환)")
    pageId: str | None = Field(None, description="Confluence pageId 힌트/잠금")
    messages: list[dict] | None = Field(None, description="대화형 호환 필드")
    sticky: bool | None = Field(None, description="이 요청에서 sticky 소스를 사용하지 않음")
    spaces: list[str] | None = Field(None, description="허용할 space 리스트(복수)")

    def to_query_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if "top_k" in d and "k" not in d:
            d["k"] = d.pop("top_k")
        if "spaceKey" in d and "space" not in d:
            d["space"] = d.pop("spaceKey")
        return d

@app.post(
    "/ask",
    summary="Ask",
    description="/ask → 내부적으로 /query 호출합니다. (벡터 검색 + 필요 시 폴백)",
)
async def ask(payload: AskPayload = Body(
    ...,
    example={"q": "NIA 사용자 시나리오", "k": 5}
)):
    """
    [중요] 여기를 Pydantic 모델로 바꿔야 OpenAPI에 '필수 q'가 반영되어
    Open WebUI가 빈 바디가 아닌 {'q': ...} 형태로 요청합니다.
    """
    # query()는 dict를 받으므로 모델을 dict로 변환
    return await query(payload.to_query_dict())

@app.post(
    "/qa",
    summary="Qa Compat",
    description="과거 호환용 엔드포인트. /ask와 동일하게 동작.",
)
async def qa_compat(payload: AskPayload = Body(
    ...,
    example={"q": "NIA 사용자 시나리오"}
)):
    """
    [중요] /qa도 동일 스키마를 쓰면 OpenAPI 상으로 동일한 요구사항이 노출됩니다.
    """
    return await query(payload.to_query_dict())


@app.post("/upload")
async def upload(file: UploadFile = File(...), overwrite: bool = Form(False)):
    ensure_dirs()
    safe_name = os.path.basename(file.filename).replace("\\", "/")
    dest = Path(UPLOAD_DIR) / safe_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File already exists: {dest}")
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    return {"saved": {"filename": safe_name, "path": str(dest), "bytes": dest.stat().st_size}}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
    parser: str = Form("auto"),
    session_id: Optional[str] = Form(None),
):
    global vectorstore, last_source   # ← 함수 제일 위에 둡니다

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    ensure_dirs()
    safe_name = os.path.basename(file.filename).replace("\\", "/")
    dest = Path(UPLOAD_DIR) / safe_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"File exists: {dest}. Set overwrite=true")

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # 로딩/청킹
    try:
        docs = load_docs_any(
            dest,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            parser=parser,
        )
        if not docs:
            raise HTTPException(400, f"no docs parsed from {dest}")
    except ValueError as ve:
        raise HTTPException(415, str(ve))
    except Exception as e:
        raise HTTPException(500, f"parse failed: {e}")

    # 업서트 (중복 방지 + 락)
    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(docs)
            after = len(vectorstore.docstore._dict)

            # ← 여기서 전역 업데이트 & sticky (전역선언은 함수 맨 위에서 이미 했음)
            last_source = _norm_source(str(dest))
            _set_sticky(last_source, session_id=session_id)
            # 세션 기반 last_source 설정
            if session_id:
                await session_manager.set_sticky(session_id, last_source)
            rebuild_domain_stats_from_index()

        return {
            "saved": {"path": str(dest), "bytes": dest.stat().st_size},
            "indexed": len(docs),
            "added": added,
            "doc_total": after,
            "parser": parser,
        }
    except Exception as e:
        raise HTTPException(500, f"index failed: {e}")


@app.post("/update")
async def update_index(payload: dict):
    global vectorstore, last_source   # ← 함수 제일 위에 둡니다

    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    # session_id 추출
    session_id = (payload or {}).get("session_id") or (payload or {}).get("sessionId")

    base = Path(UPLOAD_DIR).resolve()
    rel  = str((payload or {}).get("path") or "").replace("\\", "/")
    if not rel:
        raise HTTPException(400, "payload.path is required")

    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except Exception:
        raise HTTPException(403, "path must be inside uploads")

    if not candidate.exists():
        raise HTTPException(404, f"file not found: {candidate}")

    parser = (payload or {}).get("parser", "auto")

    # (주의) 여기서 last_source 건드리지 말 것!  ←←← 기존의 임시 sticky 라인 삭제

    # 1) 문서 로딩
    try:
        docs = load_docs_any(
            candidate,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            parser=parser,
        )
        if not docs:
            raise HTTPException(400, f"no docs parsed from {candidate}")
    except ValueError as ve:
        raise HTTPException(415, str(ve))
    except Exception as e:
        raise HTTPException(500, f"load failed: {e}")

    # 2) 업서트 + 저장 + 리트리버 갱신
    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(docs)
            after = len(vectorstore.docstore._dict)

            # 업서트 성공 후에만 최근 소스/sticky 갱신
            last_source = _norm_source(str(candidate))
            _set_sticky(last_source, session_id=session_id)
            # 세션 기반 last_source 설정
            if session_id:
                await session_manager.set_sticky(session_id, last_source)
            rebuild_domain_stats_from_index()

        return {
            "ok": True,
            "indexed": len(docs),
            "added": added,
            "doc_total": after,
            "source": str(candidate),
            "chunks": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "parser": parser,
        }
    except Exception as e:
        log.exception("update failed")
        detail = str(e) or e.__class__.__name__
        raise HTTPException(500, f"update failed: {detail}")


@app.delete("/delete")
async def delete_index(payload: Optional[dict] = None):
    global vectorstore, current_source, current_source_until, last_source
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    payload = payload or {}
    mode = payload.get("mode"); source = payload.get("source")

    if mode == "all":
        try:
            async with index_lock:
                root = Path(INDEX_DIR)
                if root.exists():
                    for child in root.iterdir():
                        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
                vectorstore = _empty_faiss()
                vectorstore.save_local(INDEX_DIR)
                _reload_retriever()
                VS.vectorstore = vectorstore; VS.retriever = retriever
                current_source = None; current_source_until = 0.0; last_source = None
            return {"deleted": "all", "doc_count": len(vectorstore.docstore._dict)}
        except Exception as e:
            raise HTTPException(500, f"delete all failed: {e}")
    
    rebuild_domain_stats_from_index()
    
    if source:
        try:
            async with index_lock:
                docstore = vectorstore.docstore._dict
                norm = _norm_source
                target = [doc_id for doc_id, doc in docstore.items()
                          if norm(str(doc.metadata.get("source",""))) == norm(source)]
                if not target:
                    return {"deleted": 0, "reason": f"no documents with source={source}"}
                vectorstore.delete(target)
                vectorstore.save_local(INDEX_DIR)
                _reload_retriever()
            if _norm_source(source) == (current_source or ""):
                current_source = None; current_source_until = 0.0
            if _norm_source(source) == (last_source or ""):
                last_source = None
            return {"deleted": len(target), "source": source, "doc_count": len(vectorstore.docstore._dict)}
        except Exception as e:
            raise HTTPException(500, f"delete by source failed: {e}")

    raise HTTPException(400, "Invalid payload. Use {'mode':'all'} or {'source':'uploads/파일.ext'}.")


@app.get("/index/stats")
def index_stats():
    """인덱스 요약(총 문서/벡터 수, 차원, 소스별/종류별 카운트)"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    try:
        ds = getattr(vectorstore, "docstore", None)
        dct = getattr(ds, "_dict", {}) if ds else {}
        total_docs = len(dct)

        by_source = Counter(str(d.metadata.get("source","")) for d in dct.values())
        by_kind   = Counter(str(d.metadata.get("kind","chunk")) for d in dct.values())

        # faiss index 통계
        idx = getattr(vectorstore, "index", None)
        dim = int(getattr(idx, "d", 0)) if idx is not None else None
        ntotal = int(getattr(idx, "ntotal", 0)) if idx is not None else None

        return {
            "doc_total": total_docs,
            "vector_total": ntotal,
            "dim": dim,
            "sources": [{"source": s or "(unknown)", "count": c} for s, c in by_source.most_common()],
            "kinds": [{"kind": k, "count": c} for k, c in by_kind.most_common()],
            "index_dir": INDEX_DIR,
            "uploads_dir": UPLOAD_DIR,
        }
    except Exception as e:
        raise HTTPException(500, f"stats failed: {e}")

@app.get("/index/sources")
def index_sources():
    """소스 파일(예: uploads/test.pdf)별 청크 수 목록"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    ds = vectorstore.docstore._dict
    by_source = Counter(str(d.metadata.get("source","")) for d in ds.values())
    items = [{"source": s or "(unknown)", "count": c} for s, c in by_source.most_common()]
    return {"total_sources": len(items), "items": items}

@app.get("/index/list")
def index_list(
    limit: int = 50,
    offset: int = 0,
    source: Optional[str] = None,
    kind: Optional[str] = None,
    contains: Optional[str] = None,
):
    """개별 청크 미리보기(페이지네이션/필터)"""
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")
    ds = vectorstore.docstore._dict

    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").replace("\u00A0"," ").strip())
    norm_contains = _norm_text(contains) if contains else None

    rows = []
    for doc_id, doc in ds.items():
        src  = str(doc.metadata.get("source",""))
        knd  = doc.metadata.get("kind", "chunk")
        text = doc.page_content or ""
        title = str(doc.metadata.get("title",""))
        
        if source and _norm_source(src) != _norm_source(source):
            continue
        if kind and knd != kind:
            continue
        if norm_contains and (norm_contains not in _norm_text(text)) and (norm_contains not in _norm_text(title)):
            continue

        rows.append({
            "id": doc_id,
            "source": src,
            "kind": knd,
            "page": doc.metadata.get("page"),
            "chars": len(text),
            "preview": (text[:200] + ("…" if len(text) > 200 else "")),
        })

    total = len(rows)
    items = rows[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": items}

@app.delete("/uploads/reset")
async def uploads_reset():
    try:
        root = Path(UPLOAD_DIR)
        if not root.exists():
            return {"status":"ok", "deleted":0}
        count = 0
        for child in root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
        return {"status":"ok", "deleted":count}
    except Exception as e:
        raise HTTPException(500, f"uploads reset failed: {e}")
    

@app.post("/query", include_in_schema=False)
async def query(payload: dict = Body(...)):
    global vectorstore, retriever, current_source, current_source_until
    fallback_attempted = False
    added = 0  # ←← 미리 초기화(하단에서 안전하게 notes에 넣거나 안 넣기 위함)

    # 1) 쿼리 추출
    q = (payload or {}).get("question") \
        or (payload or {}).get("query") \
        or (payload or {}).get("q") or ""
    if (not q.strip()) and isinstance((payload or {}).get("messages"), list):
        for m in reversed(payload["messages"]):
            if m.get("role") == "user" and m.get("content"):
                q = m["content"]; break
    if not q.strip():
        raise HTTPException(400, "question/query/q/messages is required")
    trace_id = (payload or {}).get("trace_id")
    if trace_id:
        log.info("trace_id=%s q=%r", trace_id, q[:160])

    # [ADD] session_id 추출 (동시 사용자 간섭 방지)
    session_id = (payload or {}).get("session_id") or (payload or {}).get("sessionId")
    if not session_id and trace_id:
        session_id = f"trace_{trace_id}"  # trace_id로 폴백

    # [ADD] pageId 힌트만 추출(잠금 아님)
    page_id = (payload or {}).get("pageId") or (payload or {}).get("page_id") or (payload or {}).get("pageid")
    if not page_id:
        m_pid = re.search(r"(?:pageid|page_id)\s*=\s*(\d+)", q, re.I)
        if m_pid:
            page_id = m_pid.group(1)
    page_id = str(page_id) if page_id is not None else None

    confl_hint = _has_confl_hint(q)

    # 메타태스크는 RAG/MCP 건너뜀 (맨 앞 유지)
    if re.match(r"(?is)^\s*#{3}\s*task\s*:", q):
        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "trace_id": trace_id,
            "notes": {"meta_task": True}
        }

    # === Direct-Answer 라우팅: 날짜/시간 등 상식형은 RAG/MCP 없이 즉답 (맨 앞 유지)
    if _is_datetime_question(q):
        date_str, dow_str, time_str = _now_str_kst()
        sys = (
            "너는 RAG 컨텍스트 없이도 답할 수 있는 상식형 질문에 직접 답하는 어시스턴트다. "
            f"현재 표준시({TZ_NAME})는 {date_str} {dow_str}요일 {time_str} 이다. "
            "사용자가 날짜/요일/시간을 물으면 이 값을 활용해 간결히 한국어로 답하라. "
            "사고과정은 절대 출력하지 말라."
        )
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": q},
        ]
        try:
            direct = await _call_llm(messages=msgs)
        except Exception:
            direct = f"오늘은 {date_str} {dow_str}요일입니다."

        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "source_urls": [],
            "direct_answer": direct,
            "trace_id": trace_id,
            "notes": {"routed": "no_rag", "reason": "datetime"}
        }

    # 2) k / source (여기서 k와 early source를 먼저 파악)
    k = (payload or {}).get("k") or (payload or {}).get("top_k") or 5
    try:
        k = int(k)
    except Exception:
        k = 5
    src_filter = (payload or {}).get("source")
    src_list   = (payload or {}).get("sources")
    src_set    = set(map(str, src_list)) if isinstance(src_list, list) and src_list else None

    # early forced_page_id (source에 pageId가 있으면 우선, 없으면 위에서 뽑은 page_id)
    forced_page_id = _extract_page_id(src_filter) if src_filter else None
    if not forced_page_id and page_id:
        forced_page_id = page_id

    # 3) space 힌트/allowed_spaces (→ 강제 Confluence 분기 전에 계산)
    space = (payload or {}).get("space") or (payload or {}).get("spaceKey")
    allowed_spaces = _resolve_allowed_spaces((payload or {}).get("spaces"))

    # 단일 allowed_spaces만 있으면 그걸 단일 힌트로 사용
    if not space and allowed_spaces and len(allowed_spaces) == 1:
        space = allowed_spaces[0]
    if isinstance(space, str):
        space = space.strip() or None

    # 4) 강제 Confluence 분기: 질문에 '컨플/컨플루언스' 있으면 retriever 호출 전에 바로 처리
    if confl_hint and not DISABLE_INTERNAL_MCP:
        fallback_attempted = True
        try:
            spaces_for_mcp = allowed_spaces or ([space] if space else None)
            if not spaces_for_mcp:
                ranked = _rank_spaces_for_query(q)
                spaces_for_mcp = ranked[:3] if ranked else [None]
            mcp_results = await _mcp_search_fast(
                q, forced_page_id=forced_page_id, spaces_for_mcp=spaces_for_mcp
            )
            if not forced_page_id:
                mcp_results = _filter_mcp_by_strong_tokens(mcp_results, q)
                if not mcp_results:
                    log.info("forced MCP (confluence hint) no strong match: q=%r", q)
            mcp_results = _rerank_mcp_results(q, mcp_results)
        except Exception as e:
            mcp_results = []
            log.error("forced MCP (confluence hint) failed: %s", "".join(traceback.format_exception(e)))


        # pageId 강제 필터
        if forced_page_id and mcp_results:
            mcp_results = [
                r for r in mcp_results
                if _url_has_page_id(r.get("url"), forced_page_id) or str(r.get("id") or "") == forced_page_id
            ]
        if forced_page_id and not mcp_results:
            page = await _mcp_page_text_http(forced_page_id)
            if page:
                mcp_results = [page]
            else:
                return {
                    "hits": 0,
                    "items": [],
                    "contexts": [],
                    "context_texts": [],
                    "documents": [],
                    "chunks": [],
                    "source_urls": [],
                    "direct_answer": "페이지 내용을 읽어올 수 없습니다.",
                    "notes": {
                        "mcp_text_missing": True,
                        "fallback_used": True,
                    },
                }
        if mcp_results and not _mcp_results_have_text(mcp_results):
            log.info("MCP results missing text: q=%r page_id=%s", q, forced_page_id)
            mcp_results = []

        # sticky 처리
        if mcp_results and STICKY_AFTER_MCP:
            first = mcp_results[0]
            target = first.get("url") or (f"confluence:{forced_page_id}" if forced_page_id else None)
            if target:
                _set_sticky(target)

        # 화면 items/contexts 생성 + (옵션) 인덱스 업서트
        if mcp_results:
            items, contexts, up_docs = _mcp_results_to_items(
                mcp_results, k=int(k) if isinstance(k, int) else 5
            )
            added = 0
            if MCP_WRITEBACK and up_docs:
                if MCP_WRITEBACK_TITLES_ONLY:
                    up_docs = [d for d in up_docs if (d.metadata or {}).get("kind") == "title"]
                async with index_lock:
                    added = _upsert_docs_no_dup(up_docs)
                    vectorstore.save_local(INDEX_DIR)
                    _reload_retriever()

            src_urls = _collect_source_urls_from_contexts(contexts)
            # PDF/업로드 필터 (안전망)
            items    = _filter_items_for_router(items)
            contexts = _filter_items_for_router(contexts)
            src_urls = _filter_urls(src_urls)

            return {
                "hits": len(items),
                "items": items,
                "contexts": contexts,
                "context_texts": [it["text"] for it in items],
                "documents": items,
                "chunks": items,
                "source_urls": src_urls,
                "notes": {
                    "forced_confluence": True,
                    "fallback_used": True,
                    "indexed": (added > 0),
                    **({"added": added} if added > 0 else {}),
                },
            }
        else:
            # Confluence에서 아무것도 못 찾았으면 '정보 없음'으로 종료 (로컬로 되돌리지 않음)
            return {
                "hits": 0,
                "items": [],
                "contexts": [],
                "context_texts": [],
                "documents": [],
                "chunks": [],
                "source_urls": [],
                "direct_answer": "페이지 내용을 읽어올 수 없습니다.",
                "notes": {
                    "forced_confluence": True,
                    "no_results": True
                },
            }

    # === 여기부터 로컬 RAG 경로 ===
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    # 요청 단위 sticky 비활성화 옵션
    sticky_flag = (payload or {}).get("sticky")
    ignore_sticky = (sticky_flag is False) or (isinstance(sticky_flag, str) and str(sticky_flag).lower() in ("0","false","no"))

    sticky_source: Optional[str] = None

    # sticky 적용 (유효기간 + 관련성 체크) - 세션 기반 + 전역 폴백
    if _is_title_like(q):
        sticky_source = None  # '제목형'은 특정 파일에 고정하지 않음
    else:
        # 세션 기반 sticky 우선
        if session_id and not ignore_sticky and not src_filter and not src_set:
            session_sticky = await session_manager.get_sticky(session_id)
            if session_sticky:
                if (not STICKY_STRICT) or _sticky_is_relevant(q, session_sticky):
                    if STICKY_MODE == "filter":
                        src_filter = session_sticky
                    else:  # "bonus"
                        sticky_source = session_sticky
                else:
                    await session_manager.clear_sticky(session_id)
        # 전역 sticky 폴백 (session_id 없는 경우)
        elif (not ignore_sticky) and not src_filter and not src_set:
            now = time.time()
            if current_source and now < current_source_until:
                if (not STICKY_STRICT) or _sticky_is_relevant(q, current_source):
                    if STICKY_MODE == "filter":
                        src_filter = current_source
                    else:  # "bonus"
                        sticky_source = current_source
                else:
                    current_source = None
                    current_source_until = 0.0

    # "이 파일/첨부한 파일" 지시어면 최근 업로드 파일로 고정
    global last_source
    if not src_filter and not src_set and THIS_FILE_PAT.search(q):
        # 세션 기반 last_source 우선
        if session_id:
            sess_last = await session_manager.get_last_source(session_id)
            if sess_last:
                src_filter = sess_last
                _set_sticky(sess_last, session_id=session_id)
        # 전역 last_source 폴백
        elif last_source:
            src_filter = last_source
            _set_sticky(last_source, session_id=session_id)

    # 3-A) 후보 선택은 MMR로
    try:
        docs = retriever.invoke(q)
    except Exception:
        docs = []

    # 빈 결과면 fallback (또는 예외 시에도)
    if not docs:
        try:
            docs = [d for d, _ in vectorstore.similarity_search_with_score(q, k=max(k*2, 10))]
        except Exception:
            docs = []

    # 파일명 힌트 → 동일 소스 우선
    m = re.search(r'([^\s"\'()]+\.(?:pdf|pptx|ppt|xlsx|xls|csv|txt|md))', q, re.I)
    fname = m.group(1).lower() if m else None
    if fname:
        bn = Path(fname).name.lower()
        try:
            ds = getattr(vectorstore, "docstore", None)
            all_docs = getattr(ds, "_dict", {}) if ds else {}
            cands = [
                _norm_source(str(doc.metadata.get("source","")))
                for doc in all_docs.values()
                if Path(str(doc.metadata.get("source",""))).name.lower() == bn
            ]
            if cands:
                src_filter = cands[0]
                _set_sticky(src_filter, session_id=session_id)
        except Exception:
            pass

    # 확장자 없이도 basename 매칭
    if not fname:
        def _canon_file(s: str) -> str:
            bn = Path(s).name.lower()
            for ext in ('.pdf', '.pptx', '.xlsx', '.txt', '.csv', '.md'):
                if bn.endswith(ext): bn = bn[:-len(ext)]
            return bn

        raw_tokens = re.findall(r'[\w\.\-\(\)가-힣]+', q.lower())
        norm_tokens = []
        for t in raw_tokens:
            if t.endswith("의"): t = t[:-1]
            t = t.strip()
            if t: norm_tokens.append(t)

        cand_sources = [str(d.metadata.get("source","")) for d in docs]
        bn_map = {_canon_file(s): s for s in cand_sources if s}

        hit_bn = None
        for t in norm_tokens:
            for bn in bn_map.keys():
                if t and t in bn:
                    hit_bn = bn; break
            if hit_bn: break

        if hit_bn:
            wanted = _norm_source(bn_map[hit_bn])
            docs = [d for d in docs if _norm_source(str(d.metadata.get("source",""))) == wanted]

    # 3) 소스 필터 후 상위 k
    if src_set:
        wanted = {_norm_source(str(s)) for s in src_set}
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) in wanted]
    elif src_filter:
        wanted = _norm_source(str(src_filter))
        docs = [d for d in docs if _norm_source(str((d.metadata or {}).get("source",""))) == wanted]

    if forced_page_id and PAGE_FILTER_MODE.lower() == "hard":
        def _doc_has_pid(d, pid):
            md = d.metadata or {}
            pid_md = str(md.get("page") or md.get("pageId") or "")
            url = str(md.get("url") or md.get("source") or "")
            return (pid_md == pid) or (f"pageId={pid}" in url)
        docs = [d for d in docs if _doc_has_pid(d, forced_page_id)]
    docs = docs[:k]

    # 3-B) 넉넉한 후보 풀 구성
    def key_of(d):
        md = d.metadata or {}
        return (str(md.get("source","")), md.get("page"), md.get("kind","chunk"), md.get("chunk"))

    pool_hits = []
    try:
        pairs = vectorstore.similarity_search_with_score(q, k=max(k*8, 80))
    except Exception:
        pairs = []

    base_docs = []
    try:
        base_docs = retriever.invoke(q) or []
    except Exception:
        base_docs = []

    # --- 경량 스파스(키워드) 후보 융합 ---
    want_sparse = ENABLE_SPARSE or bool(re.search(r"\b[A-Z]{2,10}\b", q)) or (len(_tokenize_query(q)) <= 3)
    if want_sparse:
        sparse_hits = _sparse_keyword_hits(q, limit=SPARSE_LIMIT, space=space)
        _apply_space_hint(sparse_hits, space)
        _apply_page_hint(sparse_hits, page_id)
        pool_hits.extend(sparse_hits)

    # L2 → 유사도 근사
    def _sim_from_dist(dist):
        try:
            d = float(dist)
            s = 1.0 - 0.5 * d
            return max(0.0, min(1.0, s))
        except Exception:
            return 0.0

    try:
        if pool_hits:
            _peek = []
            for h in pool_hits[:8]:
                md = (h.get("metadata") or {})
                _peek.append(md.get("source") or md.get("url") or "")
            log.info("pool peek sources: %s", _peek)
    except Exception:
        pass

    for d, dist in pairs:
        sim = _sim_from_dist(dist)
        md = dict(d.metadata or {})
        md["source"] = str(md.get("source",""))
        pool_hits.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": sim,
        })

    for d in base_docs:
        md = dict(d.metadata or {})
        md["source"] = str(md.get("source",""))
        pool_hits.append({
            "text": d.page_content or "",
            "metadata": md,
            "score": 0.5,
        })

    # If a source filter is provided, pull all chunks from that source
    # so we don't get stuck with only a title hit.
    if src_filter:
        try:
            wanted = _norm_source(str(src_filter))
            ds = getattr(vectorstore, "docstore", None)
            all_docs = getattr(ds, "_dict", {}).values() if ds else []
            for d in all_docs:
                md = dict(d.metadata or {})
                src = _norm_source(str(md.get("source","")))
                if src == wanted:
                    md["source"] = str(md.get("source",""))
                    pool_hits.append({
                        "text": d.page_content or "",
                        "metadata": md,
                        "score": 0.45,
                    })
        except Exception:
            pass

    # dense 풀에도 space/page 보너스
    _apply_space_hint(pool_hits, space)
    _apply_page_hint(pool_hits, page_id)
    if not _is_title_like(q):
        if _should_apply_local_bonus(q, allowed_spaces) and not _is_title_like(q):
            _apply_local_bonus(pool_hits)
    _apply_acronym_bonus(pool_hits, q)

    # sticky가 '보너스 모드'면 가산점만 반영
    if sticky_source and STICKY_MODE == "bonus":
        _apply_sticky_bonus(pool_hits, sticky_source)

    q_tokens = _tokenize_query(q)
    if q_tokens:
        for h in pool_hits:
            md = h.get("metadata") or {}
            title = (md.get("title") or "")
            if title and any(t in title for t in q_tokens):
                h["score"] = float(h.get("score") or 0.0) + (TITLE_BONUS * 0.5)

    # allowed_spaces 강제/보너스
    if allowed_spaces:
        mode = SPACE_FILTER_MODE.lower()
        allowed_set = {s.lower() for s in allowed_spaces}
        if mode == "hard":
            pool_hits = [
                h for h in pool_hits
                if ((h.get("metadata") or {}).get("space","") or "").lower() in allowed_set
            ]
        else:
            for sp in allowed_spaces:
                _apply_space_hint(pool_hits, sp)

    # === 3-D) 의도 파악
    intent = parse_query_intent(q)
    m_art = _ARTICLE_RE.search(q)
    article_no = intent.get("article_no") if m_art else None
    m_ch = _CHAPTER_RE.search(q)
    chapter_no = int(m_ch.group(1)) if m_ch else None

    if intent.get("article_no") and ("source" in (payload or {})):
        src_filter = (payload or {}).get("source")

    if src_set:
        wanted = {_norm_source(str(s)) for s in src_set}
        pool_hits = [h for h in pool_hits if _norm_source(h["metadata"].get("source","")) in wanted]
    elif src_filter:
        wanted = _norm_source(str(src_filter))
        pool_hits = [h for h in pool_hits if _norm_source(h["metadata"].get("source","")) == wanted]

    def _bonus_for_article_text(h, artno: int) -> float:
        t = re.sub(r"[ \t\r\n│|¦┃┆┇┊┋丨ㅣ]", "", h.get("text") or "")
        return 0.15 if re.search(fr"제{artno}조", t) else 0.0

    def _bonus_for_chapter_text(h, chapno: int) -> float:
        t = re.sub(r"[ \t\r\n│|¦┃┆┇┊┋丨ㅣ]", "", h.get("text") or "")
        return 0.20 if re.search(fr"제{chapno}장", t) else 0.0

    if article_no:
        for h in pool_hits:
            h["score"] = float(h.get("score") or 0.0) + _bonus_for_article_text(h, article_no)
    if chapter_no:
        for h in pool_hits:
            h["score"] = float(h.get("score") or 0.0) + _bonus_for_chapter_text(h, chapter_no)

    missing_article = False
    if article_no:
        have_meta = any((h.get("metadata") or {}).get("article_no") == article_no for h in pool_hits)
        have_text = any(_bonus_for_article_text(h, article_no) > 0 for h in pool_hits)
        missing_article = not (have_meta or have_text)

    if forced_page_id and PAGE_FILTER_MODE.lower() == "hard":
        def _hit_has_pid(h, pid):
            md  = h.get("metadata") or {}
            pid_md = str(md.get("page") or md.get("pageId") or "")
            url = str(md.get("url") or md.get("source") or "")
            return (pid_md == pid) or (f"pageId={pid}" in url)
        pool_hits = [h for h in pool_hits if _hit_has_pid(h, forced_page_id)]

    # 3-E) rerank + 의도기반 주입선택
    pool_hits = _rerank_pool_hits(q, pool_hits)
    chosen = pick_for_injection(q, pool_hits, k_default=int(k) if isinstance(k, int) else 5)
    had_pdf_ctx = any(_blocked_item(h) for h in (chosen or [])) \
           or any(_blocked_item(h) for h in (pool_hits or []))
    chosen = _coalesce_single_source(chosen, q)
    # 소스 한쪽 쏠림 방지 기본은 0(비활성) 운영에서 켜고 싶을 때만 설정으로 조정
    if PER_SOURCE_CAP > 0:
        chosen = _rebalance_by_source(chosen, k=int(k) if isinstance(k, int) else 5, per_source_cap=PER_SOURCE_CAP)
        
    if intent.get("article_no") and src_filter and not any(
        (h.get("metadata") or {}).get("article_no") == intent["article_no"] for h in chosen
    ):
        try:
            pairs2 = vectorstore.similarity_search_with_score(q, k=160)
            pool_hits2 = []
            for d, dist in pairs2:
                if _norm_source(str((d.metadata or {}).get("source",""))) != _norm_source(str(src_filter)):
                    continue
                pool_hits2.append({
                    "text": d.page_content or "",
                    "metadata": dict(d.metadata, source=str(d.metadata.get("source",""))),
                    "score": _sim_from_dist(dist),
                })
            if pool_hits2:
                chosen = pick_for_injection(q, pool_hits2, k_default=int(k) if isinstance(k, int) else 5)
        except Exception:
            pass

    # 4) 응답 생성
    items, contexts = [], []
    for h in chosen:
        md = dict(h["metadata"])
        sc = float(h.get("score") or 0.0)
        entry = {
            "text": h["text"],
            "metadata": md,
            "score": round(sc, 4),
        }
        items.append(entry)
        contexts.append({
            "text": h["text"],
            "source": md.get("source"),
            "page": md.get("page"),
            "kind": md.get("kind", "chunk"),
            "score": round(sc, 4),
            "title": md.get("title", ""),
        })

    # If source is fixed but only title/summary are selected, fall back to chunks from that source.
    # [FIX] 유사도 기반 청크 선택으로 개선 (고정 스코어 0.35 → 실제 유사도)
    if src_filter and (not items or all((it.get("metadata") or {}).get("kind") in ("title", "summary") for it in items)):
        try:
            wanted = _norm_source(str(src_filter))

            # [NEW] 유사도 검색으로 해당 소스의 청크 가져오기
            try:
                # vectorstore에서 유사도 검색 (k*3으로 넉넉하게)
                search_results = vectorstore.similarity_search_with_score(q, k=max(k*3, 15))
                alt = []
                for d, dist in search_results:
                    md = dict(d.metadata or {})
                    if _norm_source(str(md.get("source",""))) != wanted:
                        continue
                    if md.get("kind") in ("title", "summary"):
                        continue
                    # 유사도 점수 계산 (L2 거리 → 유사도)
                    sim = 1.0 - 0.5 * float(dist)
                    sim = max(0.0, min(1.0, sim))
                    alt.append({
                        "text": d.page_content or "",
                        "metadata": md,
                        "score": sim,
                    })

                # 유사도순 정렬 후 상위 k개
                alt.sort(key=lambda x: x["score"], reverse=True)
                alt = alt[:k]

            except Exception:
                # 유사도 검색 실패 시 기존 방식 (전체 스캔)
                ds = getattr(vectorstore, "docstore", None)
                all_docs = getattr(ds, "_dict", {}).values() if ds else []
                alt = []
                for d in all_docs:
                    md = dict(d.metadata or {})
                    if _norm_source(str(md.get("source",""))) != wanted:
                        continue
                    if md.get("kind") in ("title", "summary"):
                        continue
                    alt.append({
                        "text": d.page_content or "",
                        "metadata": md,
                        "score": 0.35,
                    })
                alt = alt[:k]

            if alt:
                items = []
                contexts = []
                for h in alt:
                    md = dict(h["metadata"])
                    sc = float(h.get("score") or 0.0)
                    items.append({"text": h["text"], "metadata": md, "score": round(sc, 4)})
                    contexts.append({
                        "text": h["text"],
                        "source": md.get("source"),
                        "page": md.get("page"),
                        "kind": md.get("kind", "chunk"),
                        "score": round(sc, 4),
                        "title": md.get("title", ""),
                    })
        except Exception:
            pass
    
    # PDF/업로드 필터
    items    = _filter_items_for_router(items)
    contexts = _filter_items_for_router(contexts)

    try:
        log.info(
            "Q=%r src_filter=%r sticky=%r page_id=%r pool=%d chosen=%d by_section=%s",
            q, src_filter, current_source, forced_page_id, len(pool_hits), len(items),
            dict(Counter((h.get('metadata') or {}).get('section_index', -1) for h in chosen))
        )
    except Exception:
        pass

    def _query_tokens(q: str) -> List[str]:
        q = _preseg_stop_phrases(_basic_normalize(q))
        raw = re.findall(r"[가-힣A-Za-z0-9]{2,}", q)
        toks = []
        for t in raw:
            t = _STOP_SUFFIX_RE.sub("", t)
            t = _strip_josa(t)
            t = _apply_canon_map(t)
            t = _collapse_korean_compounds(t)
            if t and t not in _K_STOP:
                toks.append(t)
        return list(dict.fromkeys(toks))[:8]

    ctx_all = "\n".join(c["text"] for c in contexts)
    tokens = _query_tokens(q)
    core, acr = _split_core_and_acronyms(tokens)

    core_hit = any(t in ctx_all for t in core)
    titles_meta = " ".join(
         f"{(h.get('metadata') or {}).get('title','')} {(h.get('metadata') or {}).get('source','')}"
         for h in items
     )

    acronym_hit = True if not acr else any(a in ctx_all or a in titles_meta for a in acr)
    anchors = _anchor_tokens_from_query(q)
    blob_norm = _norm_kr(ctx_all + " " + titles_meta)
    anchor_hit = (not anchors) or any(_norm_kr(a) in blob_norm for a in anchors)

    anchor_miss = bool(anchors) and not anchor_hit
    acronym_miss = bool(acr) and not acronym_hit

    local_ok = _has_local_hits(items) or _has_local_hits(pool_hits)
    # score/coverage 기반 부족 판정(옵션)
    top1 = top2 = 0.0
    if items:
        scores = sorted([float(it.get("score") or 0.0) for it in items], reverse=True)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
    covered = 0
    if core:
        for t in core:
            if t in ctx_all or t in titles_meta:
                covered += 1
    core_cover = (covered / len(core)) if core else 1.0
    score_low = RAG_SCORE_GATES and (top1 < RAG_TOP1_MIN)
    gap_low = RAG_SCORE_GATES and ((top1 - top2) < RAG_TOP_GAP_MIN)
    cover_low = RAG_SCORE_GATES and (core_cover < RAG_CORE_COVER_MIN)

    if local_ok:
        NEED_FALLBACK = (len(items) == 0) or missing_article or anchor_miss or acronym_miss
    else:
        NEED_FALLBACK = (
            (len(items) == 0) or
            (len(pool_hits) < max(10, k*2)) or
            missing_article or
            acronym_miss or
            anchor_miss
        )
    if score_low or gap_low or cover_low:
        NEED_FALLBACK = True

    if (chapter_no or article_no) and items:
        NEED_FALLBACK = False

    client_spaces = (payload or {}).get("spaces")
    allowed_spaces = _resolve_allowed_spaces(client_spaces)

    reasons = []
    if len(items) == 0: reasons.append("no_items")
    if len(pool_hits) < max(10, k*2): reasons.append("small_pool")
    if missing_article: reasons.append("missing_article")
    if (acr and not acronym_hit): reasons.append("acronym_miss")
    if (anchors and not anchor_hit): reasons.append("anchor_miss")
    if score_low: reasons.append("low_score")
    if gap_low: reasons.append("low_gap")
    if cover_low: reasons.append("low_coverage")
    log.info("fallback_check reasons=%s local_hits=%s", reasons, local_ok)

    pid_miss = False
    if forced_page_id:
        def _hit_has_pid(h, pid):
            md  = (h.get("metadata") or {})
            pid_md = str(md.get("page") or md.get("pageId") or "")
            url = str(md.get("url") or md.get("source") or "")
            return (pid_md == pid) or (f"pageId={pid}" in url)

        pid_miss = (not any(_hit_has_pid(h, forced_page_id) for h in (items or []))) \
                and (not any(_hit_has_pid(h, forced_page_id) for h in (pool_hits or [])))

        if pid_miss:
            NEED_FALLBACK = True
            reasons.append("pid_miss")

    if "anchor_miss" in reasons and reasons == ["anchor_miss"]:
        NEED_FALLBACK = (not local_ok) or _should_use_mcp(q, allowed_spaces, space, reasons, local_ok)
        log.info("MCP %s: anchor_miss only (local_ok=%s, domain_gate=%s)",
                "allowed" if NEED_FALLBACK else "skipped", local_ok,
                _should_use_mcp(q, client_spaces, space, reasons, local_ok))

    allow_reasons = ("no_items", "small_pool", "missing_article", "pid_miss")
    allow_fallback = (
        any(r in reasons for r in allow_reasons) or
        ("anchor_miss" in reasons and not local_ok) or
        ("acronym_miss" in reasons)
    )
    explicit_src_filter = bool(src_filter) and not forced_page_id
    if explicit_src_filter and allow_fallback:
        allow_fallback = False
        if NEED_FALLBACK:
            NEED_FALLBACK = False
            log.info("MCP fallback skipped due to explicit src_filter=%r", src_filter)
    
    if NEED_FALLBACK and not _should_use_mcp(q, allowed_spaces, space, reasons, local_ok):
        NEED_FALLBACK = False
        log.info("MCP skipped by domain gate for %r", q)

    if NEED_FALLBACK and not DISABLE_INTERNAL_MCP and allow_fallback:
        fallback_attempted = True
        mcp_results = []
        try:
            log.info("MCP fallback (fast): q=%r", q)
            spaces_for_mcp = allowed_spaces or ([space] if space else None)
            if not spaces_for_mcp:
                ranked = _rank_spaces_for_query(q)
                spaces_for_mcp = ranked[:3] if ranked else [None]
            mcp_results = await _mcp_search_fast(
                q, forced_page_id=forced_page_id, spaces_for_mcp=spaces_for_mcp
            )
            if not forced_page_id:
                mcp_results = _filter_mcp_by_strong_tokens(mcp_results, q)
                if not mcp_results:
                    log.info("MCP fallback no strong match: q=%r", q)
            mcp_results = _rerank_mcp_results(q, mcp_results)
        except Exception as e:
            log.error("MCP fallback fast failed: %s", "".join(traceback.format_exception(e)))

        if forced_page_id and mcp_results:
            mcp_results = [
                r for r in mcp_results
                if _url_has_page_id(r.get("url"), forced_page_id) or str(r.get("id") or "") == forced_page_id
            ]
        if forced_page_id and not mcp_results:
            page = await _mcp_page_text_http(forced_page_id)
            if page:
                mcp_results = [page]

        if mcp_results and not _mcp_results_have_text(mcp_results):
            log.info("MCP results missing text: q=%r page_id=%s", q, forced_page_id)
            mcp_results = []
            return {
                "hits": 0,
                "items": [],
                "contexts": [],
                "context_texts": [],
                "documents": [],
                "chunks": [],
                "source_urls": [],
                "direct_answer": "페이지 내용을 읽어올 수 없습니다.",
                "notes": {
                    "mcp_text_missing": True,
                    "fallback_used": True,
                },
            }

        if mcp_results and STICKY_AFTER_MCP:
            first = mcp_results[0]
            target = first.get("url") or (f"confluence:{forced_page_id}" if forced_page_id else None)
            if target:
                _set_sticky(target)

        if mcp_results:
            items, contexts, up_docs = _mcp_results_to_items(
                mcp_results, k=int(k) if isinstance(k, int) else 5
            )
            added = 0
            if MCP_WRITEBACK and up_docs:
                if MCP_WRITEBACK_TITLES_ONLY:
                    up_docs = [d for d in up_docs if (d.metadata or {}).get("kind") == "title"]
                async with index_lock:
                    added = _upsert_docs_no_dup(up_docs)
                    vectorstore.save_local(INDEX_DIR)
                    _reload_retriever()

            try:
                if mcp_results and (mcp_results[0].get("url") or mcp_results[0].get("id")):
                    want_sticky = False
                    if forced_page_id or THIS_FILE_PAT.search(q) or re.search(r'([^\s"\'()]+\.(?:pdf|pptx|ppt|xlsx|xls|csv|txt|md|docx))', q, re.I):
                        want_sticky = True
                    if STICKY_AFTER_MCP and want_sticky:
                        try:
                            first = mcp_results[0]
                            _set_sticky(first.get("url") or f"confluence:{first.get('id')}")
                        except Exception:
                            pass
            except Exception:
                pass

    # 장/조 질의면 필터 스킵
    if items and not (chapter_no or article_no) and not explicit_src_filter:
        if not THIS_FILE_PAT.search(q):
            ctx_all_final = "\n".join(c["text"] for c in contexts)
            titles_meta_final = " ".join(
                f"{(h.get('metadata') or {}).get('title','')} {(h.get('metadata') or {}).get('source','')}"
                for h in items
            )
            core_final = [t for t in _query_tokens(q) if not ACRONYM_RE.match(t)]
            blob_norm_final = _norm_kr(ctx_all_final + " " + titles_meta_final)
            if core_final and not any(_norm_kr(t) in blob_norm_final for t in core_final):
                items, contexts = [], []

            if items:
                anchors = _anchor_tokens_from_query(q) + re.findall(r"[A-Z]{2,5}", q)
                if anchors and not any(_norm_kr(a) in blob_norm_final for a in anchors):
                    items, contexts = [], []

        # 비었을 때의 보조 폴백도 fast 버전으로
        if not items and (not explicit_src_filter) and not DISABLE_INTERNAL_MCP and _should_use_mcp(q, client_spaces, space, reasons, local_ok):
            fallback_attempted = True
            spaces_for_mcp = allowed_spaces or ([space] if space else None)
            if not spaces_for_mcp:
                ranked = _rank_spaces_for_query(q)
                spaces_for_mcp = ranked[:3] if ranked else [None]
            mcp_results = await _mcp_search_fast(
                q, forced_page_id=forced_page_id, spaces_for_mcp=spaces_for_mcp
            )
            if not forced_page_id:
                mcp_results = _filter_mcp_by_strong_tokens(mcp_results, q)
            mcp_results = _rerank_mcp_results(q, mcp_results)
            if forced_page_id and mcp_results:
                mcp_results = [r for r in mcp_results
                            if _url_has_page_id(r.get("url"), forced_page_id)
                            or str(r.get("id") or "") == forced_page_id]

            if mcp_results:
                items, contexts, up_docs = _mcp_results_to_items(mcp_results, k=int(k) if isinstance(k, int) else 5)

                added = 0
                if MCP_WRITEBACK and up_docs:
                    if MCP_WRITEBACK_TITLES_ONLY:
                        up_docs = [d for d in up_docs if (d.metadata or {}).get("kind") == "title"]
                    async with index_lock:
                        added = _upsert_docs_no_dup(up_docs)
                        vectorstore.save_local(INDEX_DIR)
                        _reload_retriever()

                if STICKY_AFTER_MCP:
                    first = mcp_results[0]
                    target = first.get("url") or (f"confluence:{forced_page_id}" if forced_page_id else None)
                    if target:
                        _set_sticky(target)

    base_notes = {"missing_article": missing_article, "article_no": article_no}
    if trace_id:
        base_notes["trace_id"] = trace_id
    if fallback_attempted:
        base_notes["fallback_used"] = True
        base_notes["indexed"] = (added > 0)
        if added > 0:
            base_notes["added"] = added

    if not items:
        base_notes["low_relevance"] = True  # ← 컨텍스트 사용 부적합 신호를 명시

    if not items:
        return {
            "hits": 0,
            "items": [],
            "contexts": [],
            "context_texts": [],
            "documents": [],
            "chunks": [],
            "source_urls": [],
            "direct_answer": "주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다",
            "notes": base_notes | {"low_relevance": True},
        }


    src_urls = _collect_source_urls_from_contexts(contexts)
    src_urls = _filter_urls(src_urls)

    return {
        "hits": len(items),
        "items": items,
        "contexts": contexts,
        "context_texts": [it["text"] for it in items],
        "documents": items,
        "chunks": items,
        "source_urls": src_urls,
        "notes": base_notes | {"had_pdf_ctx": bool(had_pdf_ctx)},
    }


# ------- helper: chunk text -------
def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []
    out = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + int(size))
        out.append(text[i:j])
        if j == n:
            break
        i = j - int(overlap)
        if i < 0:
            i = 0
    return out

_URL_CANON_RE = re.compile(r"(pageId=\d+)")

def _canon_url(u: Optional[str], pid: Optional[str] = None) -> str:
    if not u and not pid:
        return ""
    s = (u or "").split("#")[0].strip().rstrip("/")
    m = _URL_CANON_RE.search(s)
    if m:
        base = s.split("?", 1)[0]
        return f"{base}?{m.group(1)}"
    if pid:
        mhost = _URL_HOST_RE.match(s)
        host  = mhost.group(1) if mhost else None
        if host:
            base = f"https://{host}/pages/viewpage.action"
        elif CONFLUENCE_BASE_URL:
            base = f"{CONFLUENCE_BASE_URL}/pages/viewpage.action"
        else:
            base = "/pages/viewpage.action"
        return f"{base}?pageId={pid}"
    return s

# ------- helper: collect source urls -------
def _collect_source_urls(items: list[dict]) -> list[str]:
    """items[*].metadata.url or metadata.source에서 고유 URL만 추출"""
    urls: list[str] = []
    for it in items or []:
        md = (it.get("metadata") or {})
        url = md.get("url") or md.get("source")
        if url and url not in urls:
            urls.append(url)
    return urls


### [FIX] 로컬 히트 감지: source와 url 모두 확인 + 경로 정규화
def _has_local_hits(entries) -> bool:
    for it in entries or []:
        md = it.get("metadata") or {}
        src = str(md.get("source") or md.get("url") or "")
        if _is_local_source(src):
            return True
    return False

@app.post("/documents/upsert")
async def documents_upsert(payload: dict = Body(...)):
    """
    payload = {"docs":[{"id":"confluence:<page_id>",
                        "text":"<long text>",
                        "metadata":{"source":"confluence","title":"...","url":"...","space":"ENG"}}]}
    """
    global vectorstore
    if vectorstore is None:
        raise HTTPException(500, "vectorstore is not ready.")

    docs = (payload or {}).get("docs")
    if not isinstance(docs, list):
        raise HTTPException(400, "docs(list) is required")

    to_add = []
    for d in docs:
        text = (d.get("text") or "").strip()
        # meta = d.get("metadata") or {}
        meta_in = (d.get("metadata") or d.get("meta") or {})
        if not text:
            continue

        title = (meta_in.get("title") or "").strip()
        url   = (meta_in.get("url") or meta_in.get("source") or "").strip()
        space = (meta_in.get("space") or meta_in.get("spaceKey") or "").strip()
        source = url if url else str(d.get("id") or "confluence")

        doc_meta = dict(meta_in)               # ← 원본 메타 유지
        doc_meta.setdefault("source", source)
        doc_meta.setdefault("title", title)
        doc_meta.setdefault("space", space)
        doc_meta.setdefault("kind", "chunk")
        # [ADD] pageId가 오면 'page'에도 복사(중복 방지 ID 계산에 유리)
        if "page" not in doc_meta and doc_meta.get("pageId"):
            doc_meta["page"] = doc_meta.get("pageId")

        for c in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            to_add.append(Document(
                page_content=c,
                metadata=doc_meta    # ← [CHANGE] 재구성하지 말고 보강된 메타 그대로 저장
            ))

    if not to_add:
        try:
            n = len(vectorstore.docstore._dict)
        except Exception:
            n = None
        return {"added": 0, "doc_total": n}

    try:
        async with index_lock:
            before = len(vectorstore.docstore._dict)
            added = _upsert_docs_no_dup(to_add)
            after  = len(vectorstore.docstore._dict)
        return {"added": added, "doc_total": after}
    except Exception as e:
        raise HTTPException(500, f"documents upsert failed: {e}")

# ← [ADD] 호환용 별칭: 브리지에서 /upsert, /v1/ingest, /v1/upsert 도 시도하므로 모두 수용
@app.post("/upsert")
async def upsert_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

@app.post("/v1/ingest")
async def v1_ingest_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

@app.post("/v1/upsert")
async def v1_upsert_alias(payload: dict = Body(...)):
    return await documents_upsert(payload)

app.include_router(smart_router)

@app.post("/v1/chat/completions")
async def v1_chat(payload: dict = Body(...)):
    import time as _time

    if not CONFLUENCE_BASE_URL:
        log.warning("CONFLUENCE_BASE_URL 미설정: http(s) 링크가 아닌 /pages/... 로 생성될 수 있습니다. "
                    "출처 렌더링이 제거될 수 있으니 운영에서는 반드시 설정하세요.")

    # 마지막 user 메시지
    q = ""
    for m in payload.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            q = m["content"]
    if not q:
        q = payload.get("prompt") or ""
    if not q.strip():
        raise HTTPException(400, "user message required")

    # 1) 내부 /query 호출
    page_id = payload.get("pageId") or payload.get("page_id")
    space   = payload.get("space") or payload.get("spaceKey")
    source  = payload.get("source")
    sources = payload.get("sources")
    spaces  = payload.get("spaces")

    r = await query({
        "question": q,
        "pageId": page_id,
        "space": space,
        "source": source,
        "sources": sources,
        "spaces": spaces,
        "sticky": False,
    })

    # notes를 '가장 먼저' 확보 (이후 로직에서 참조)
    notes = r.get("notes", {}) or {}

    use_contexts = bool(r.get("contexts")) and not notes.get("low_relevance", False)

    def _has_uploads_in_response(resp: dict) -> bool:
        # source_urls 에 업로드/ PDF 흔적이 있으면 즉시 True
        for u in (resp.get("source_urls") or []):
            if _looks_blocked_source(u) or _PDF_EXT_RE.search(str(u or "")):
                return True
        # contexts / items 메타데이터에도 업로드/ PDF 흔적 있는지 확인
        for coll in (resp.get("contexts") or [], resp.get("items") or []):
            if isinstance(coll, list):
                for it in coll:
                    md = (it.get("metadata") or {})
                    src = str(md.get("source") or md.get("url") or "")
                    mime = (md.get("mimetype") or md.get("mime") or md.get("content_type") or "")
                    if _looks_blocked_source(src) or _PDF_EXT_RE.search(src) or _is_pdf_mime(mime):
                        return True
        return False

    # 2) direct_answer 우선 사용 (예: 날짜/시간 즉답 라우팅)
    content = r.get("direct_answer")

    # 3) 이 응답에서 컨텍스트를 쓸지/말지 결정
    pdf_related = bool(notes.get("had_pdf_ctx")) \
        or _is_pdf_related_response(r) \
        or bool(_PDF_EXT_RE.search(q or "")) \
        or bool(THIS_FILE_PAT.search(q or ""))

    # 안전망: 위 조건이 False라도 응답 안에 업로드/ PDF 흔적이 있으면 True로 확정
    if not pdf_related and _has_uploads_in_response(r):
        pdf_related = True

    # 디버그 로그로 실제 판정값 확인
    log.info("v1_chat gate: pdf_related=%s had_pdf_ctx=%s use_contexts=%s src_urls=%s",
            pdf_related, notes.get("had_pdf_ctx"), bool(r.get("contexts")) and not notes.get("low_relevance", False),
            r.get("source_urls"))

    # 4) content가 없으면 컨텍스트 요약 또는 일반지식 경로로 생성
    if not content:
        if use_contexts:
            ctx = "\n\n---\n\n".join(c.get("text", "") for c in r.get("contexts", [])[:6]).strip()[:8000]
            if ctx:
                sys = lang_wrap(
                    "사고과정을 출력하지 말고, 아래 컨텍스트에 **근거한 사실만** 한국어로 답하라. "
                    "질문에 '제 N장' 또는 '제 N조'가 있으면, 컨텍스트에서 해당 장/조를 먼저 **그대로 인용(>)**하고, "
                    "다음 줄에 한 문장 요약을 붙여라. 컨텍스트에 없는 내용은 '컨텍스트에 해당 정보가 없습니다'라고 말하라.\n\n"
                    "컨텍스트:\n" + ctx
                )
                msgs = [{"role": "system", "content": sys}, {"role": "user", "content": q}]
                content = await _call_llm(messages=msgs, max_tokens=700, temperature=0.2)
            else:
                use_contexts = False  # 컨텍스트가 비정상적으로 비면 일반지식으로 폴백

        if not use_contexts:
            sys = lang_wrap(
                "너는 내부 문서를 인용하지 않고도 답할 수 있는 일반 지식 질문에 답하는 어시스턴트다. "
                "사고과정은 출력하지 말고, 한국어로 간결하고 정확하게 답하라. 출처나 링크는 붙이지 마라."
            )
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": q}]
            content = await _call_llm(messages=msgs, max_tokens=700, temperature=0.2)

    # 5) 최종 출력 후처리: 생각 제거 → (조건적) 인라인 출처 제거
    if content:
        try:
            content = drop_think(content)
        except Exception:
            pass

        sources_for_strip = r.get("source_urls") or []
        # 로컬 업로드/PDF 맥락이면 'auto' 모드 트리거를 위해 업로드 형태를 주입
        if notes.get("had_pdf_ctx") or _is_pdf_related_response(r) or THIS_FILE_PAT.search(q) or _FILE_EXT_RE.search(q):
            sources_for_strip = ["uploads/__local__.pdf"]

        content = _maybe_strip_citations(content, q, sources_for_strip)

    # 6) 화면용 '출처:' 블록은 (컨텍스트 사용 & PDF 관련 아님)일 때만 추가
    if SHOW_SOURCE_BLOCK != "never" and use_contexts and not pdf_related:
        allowed_list = _filter_urls_by_host(r.get("source_urls", []) or [])
        allowed_http = [u for u in allowed_list if isinstance(u, str) and u.startswith(("http://", "https://"))]
        if allowed_http:
            src_block = "\n\n출처:\n" + "\n".join(f"- {u}" for u in allowed_http)
            content = (content or "") + src_block

    # 7) OpenAI 호환 응답
    return {
        "object": "chat.completion",
        "model": "qwen3-30b-a3b-fp8-router",
        "created": int(_time.time()),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "notes": r.get("notes", {}),
    }
