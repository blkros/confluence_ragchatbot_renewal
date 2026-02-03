# rag-router/app.py
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os, httpx, time, uuid, re, math, unicodedata, asyncio, json
from html import unescape
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import lru_cache

RAG = os.getenv("RAG_PROXY_URL", "http://rag-proxy:8080")
OPENAI = os.getenv("OPENAI_URL", "http://172.16.10.168:9993/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "/model/Qwen2.5-14B-Instruct")
ROUTER_MODEL_ID = os.getenv("ROUTER_MODEL_ID", "qwen3-30b-a3b-fp8-router")
TZ = os.getenv("ROUTER_TZ", "Asia/Seoul")

# [ADD] Clarification 상태 저장 (세션별)
# key: session_id (trace_id 사용), value: {"candidates": [...], "timestamp": float}
_CLARIFICATION_SESSIONS = {}
_CLARIFICATION_TIMEOUT = 300  # 5분
_NUM_ONLY_LINE = re.compile(r'(?m)^\s*(\d{1,3}(?:,\d{3})*|\d+)\s*$')
_FILE_HINT_RE = re.compile(
    r"(?:file|document|docx|doc|word|pdf|pptx|ppt|xlsx|excel|csv|sheet|slide|"
    r"\uCCA8\uBD80|\uD30C\uC77C|\uBB38\uC11C|\uC790\uB8CC|\uC5C5\uB85C\uB4DC|"
    r"\uC5D1\uC140|\uD30C\uC6CC\uD3EC\uC778\uD2B8|\uD53C\uD53C\uD2F0)",
    re.I,
)
_LOCAL_SRC_RE = re.compile(r"/uploads/|\\uploads\\", re.I)
_CONF_HOST_RE = re.compile(r"https?://[^/]*confluence[^/]*", re.I)
_CONF_HINT_RE = re.compile(r"(confluence|컨플|컨플루언스)", re.I)
_PAGEID_HINT_RE = re.compile(r"pageid\s*=\s*\d+", re.I)
_CONTEXT_CUT_RE = re.compile(
    r"\s*[-\u2013\u2014]{2,}\s*\[?context\]?\s*.*$|\s*[-\u2013\u2014]{2,}\s*\[?\uCEE8\uD14D\uC2A4\uD2B8\]?\s*.*$",
    re.I | re.S,
)

def _openai_headers() -> dict:
    if not OPENAI_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}

ROUTER_STRICT_RAG = (os.getenv("ROUTER_STRICT_RAG", "1").lower() not in ("0","false","no"))
ANSWER_MIN_OVERLAP = float(os.getenv("ROUTER_ANSWER_MIN_OVERLAP", "0.12"))


ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "2048"))
ANSWER_MODE = os.getenv("ROUTER_ANSWER_MODE", "auto")
BULLETS_MAX = int(os.getenv("ROUTER_BULLETS_MAX", "15"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "8000"))
_BULLET_HINTS = os.getenv(
    "ROUTER_AUTO_BULLET_HINTS",
    "\uC694\uC57D \uBAA9\uCC28 \uBAA9\uB85D \uB9AC\uC2A4\uD2B8 \uBD88\uB9BF bullet "
    "\uCCB4\uD06C\uB9AC\uC2A4\uD2B8 \uC7A5\uB2E8\uC810 \uBE44\uAD50 \uD655\uC778 "
    "\uD575\uC2EC todo \uD574\uC57C\uD560\uC77C"
).split()
_PARA_HINTS = os.getenv(
    "ROUTER_AUTO_PARA_HINTS",
    "\uC124\uBA85 \uC5B4\uB5A4 \uC124\uBA85\uC774\uC57C \uC124\uBA85\uD574\uC918 "
    "\uAC1C\uC694 \uBB34\uC5C7 \uBB50\uC57C \uBB50\uC9C0 \uD574\uC918 "
    "\uC790\uC138\uD788 \uC5B4\uB5A4 \uC815\uC758 \uAC1C\uC694 \uBCF8\uBB38 \uBB38\uB2E8 \uC11C\uC220\uD615"
).split()

# [추가] 제목 접두어 완전 비활성(기본 ""), 필요시 환경변수로 켜도 됨
HEADING = os.getenv("ROUTER_HEADING", "")

# [추가] 출처 표시 on/off 및 최대 개수
ROUTER_SHOW_SOURCES = (os.getenv("ROUTER_SHOW_SOURCES", "1").lower() not in ("0","false","no"))
ROUTER_SOURCES_MAX  = int(os.getenv("ROUTER_SOURCES_MAX", "5"))
ROUTER_SHOW_CONTEXT_LABEL = (os.getenv("ROUTER_SHOW_CONTEXT_LABEL", "1").lower() not in ("0","false","no"))
ROUTER_DEBUG = (os.getenv("ROUTER_DEBUG", "0").lower() not in ("0","false","no"))
ROUTER_QA_MIN_CTX_LEN = int(os.getenv("ROUTER_QA_MIN_CTX_LEN", "120"))
ROUTER_TITLE_EXPAND = (os.getenv("ROUTER_TITLE_EXPAND", "1").lower() not in ("0","false","no"))
ROUTER_TITLE_EXPAND_TOPN = int(os.getenv("ROUTER_TITLE_EXPAND_TOPN", "3"))
ROUTER_TITLE_EXPAND_K = int(os.getenv("ROUTER_TITLE_EXPAND_K", "60"))
ROUTER_TITLE_EXPAND_KINDS = set(
    k.strip() for k in os.getenv("ROUTER_TITLE_EXPAND_KINDS", "title,summary,section").split(",") if k.strip()
)
ROUTER_INGEST_WAIT_SEC = float(os.getenv("ROUTER_INGEST_WAIT_SEC", "8"))
ROUTER_INGEST_WAIT_INTERVAL = float(os.getenv("ROUTER_INGEST_WAIT_INTERVAL", "1.0"))
ROUTER_UPLOADS_DIR = os.getenv("ROUTER_UPLOADS_DIR", "/data/uploads")
ROUTER_USER_MAX_CHARS = int(os.getenv("ROUTER_USER_MAX_CHARS", "1200"))
MODEL_LIMIT_TOKENS = int(os.getenv("ROUTER_MODEL_LIMIT_TOKENS", "8192"))
SAFETY_MARGIN = int(os.getenv("ROUTER_CTX_SAFETY_MARGIN", "512"))
OUTPUT_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "2048"))
ALLOW_LLM_FALLBACK = (os.getenv("ROUTER_ALLOW_LLM_FALLBACK", "1").lower() not in ("0","false","no"))
FOLLOWUP_MAX_CHARS = int(os.getenv("ROUTER_FOLLOWUP_MAX_CHARS", "120"))
ROUTER_TRI_STATE = (os.getenv("ROUTER_TRI_STATE", "0").lower() not in ("0","false","no"))
ROUTER_TRI_STATE_ENFORCE = (os.getenv("ROUTER_TRI_STATE_ENFORCE", "0").lower() not in ("0","false","no"))
ROUTER_LLM_ROUTE = (os.getenv("ROUTER_LLM_ROUTE", "0").lower() not in ("0","false","no"))
ROUTER_LLM_ROUTE_STRICT = (os.getenv("ROUTER_LLM_ROUTE_STRICT", "0").lower() not in ("0","false","no"))
ROUTER_LLM_ROUTE_TIMEOUT = float(os.getenv("ROUTER_LLM_ROUTE_TIMEOUT", "3.0"))
ROUTER_LLM_ROUTE_SCORE = (os.getenv("ROUTER_LLM_ROUTE_SCORE", "1").lower() not in ("0","false","no"))
ROUTER_LLM_ROUTE_SCORE_THRESHOLD = int(os.getenv("ROUTER_LLM_ROUTE_SCORE_THRESHOLD", "7"))
ROUTER_LLM_ROUTE_REQUIRED_THRESHOLD = int(os.getenv("ROUTER_LLM_ROUTE_REQUIRED_THRESHOLD", "9"))
ROUTER_PRECHECK_RAG = (os.getenv("ROUTER_PRECHECK_RAG", "0").lower() not in ("0","false","no"))
ROUTER_PRECHECK_K = int(os.getenv("ROUTER_PRECHECK_K", "3"))
ROUTER_PRECHECK_TIMEOUT = float(os.getenv("ROUTER_PRECHECK_TIMEOUT", "2.0"))
ROUTER_PRECHECK_CONNECT_TIMEOUT = float(os.getenv("ROUTER_PRECHECK_CONNECT_TIMEOUT", "1.5"))
ROUTER_PRECHECK_RETRIES = int(os.getenv("ROUTER_PRECHECK_RETRIES", "1"))
ROUTER_REWRITE = (os.getenv("ROUTER_REWRITE", "0").lower() not in ("0","false","no"))
ROUTER_REWRITE_TURNS = int(os.getenv("ROUTER_REWRITE_TURNS", "6"))
ROUTER_FORCE_KOREAN = (os.getenv("ROUTER_FORCE_KOREAN", "1").lower() not in ("0","false","no"))

# === relevance gate ===
_KO_EN_TOKEN = re.compile(r"[A-Za-z0-9]+|[\uAC00-\uD7A3]{2,}")
ROUTER_USE_HINTS = os.getenv("ROUTER_USE_HINTS", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)
_RAG_REQUIRED_HINTS_RAW = os.getenv("ROUTER_RAG_REQUIRED_HINTS", "").strip()
_RAG_PREFERRED_HINTS_RAW = os.getenv("ROUTER_RAG_PREFERRED_HINTS", "").strip()
_RAG_REQUIRED_HINTS = (
    re.compile(_RAG_REQUIRED_HINTS_RAW, re.I) if _RAG_REQUIRED_HINTS_RAW else None
)
_RAG_PREFERRED_HINTS = (
    re.compile(_RAG_PREFERRED_HINTS_RAW, re.I) if _RAG_PREFERRED_HINTS_RAW else None
)
_KO_CHAR_RE = re.compile(r"[\uAC00-\uD7A3]")
_CJK_CHAR_RE = re.compile(r"[\u4E00-\u9FFF]")

_BASE_SYNONYMS: dict[str, list[str]] = {}
_BASE_ALIASES: dict[str, list[str]] = {}

def _load_json_map(env_json: str, env_path: str) -> dict:
    raw = os.getenv(env_json, "").strip()
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return {}
    path = os.getenv(env_path, "").strip()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _normalize_map(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        elif isinstance(v, str):
            out[str(k)] = [v]
        else:
            out[str(k)] = [str(v)]
    return out

def _merge_map(base: dict, override: dict, merge: bool) -> dict:
    if not override:
        return base
    if not merge:
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = sorted(set(out.get(k, []) + v))
    return out

_MERGE_SYNONYMS = os.getenv("ROUTER_SYNONYMS_MERGE", "1").lower() not in ("0", "false", "no")
_MERGE_ALIASES = os.getenv("ROUTER_ALIASES_MERGE", "1").lower() not in ("0", "false", "no")
_MERGE_STOPWORDS = os.getenv("ROUTER_STOPWORDS_MERGE", "1").lower() not in ("0", "false", "no")
_MERGE_FOCUS_KEYWORDS = os.getenv("ROUTER_FOCUS_KEYWORDS_MERGE", "1").lower() not in ("0", "false", "no")

_syn_raw = _normalize_map(_load_json_map("ROUTER_SYNONYMS_JSON", "ROUTER_SYNONYMS_PATH"))
_alias_raw = _normalize_map(_load_json_map("ROUTER_ALIASES_JSON", "ROUTER_ALIASES_PATH"))
SYNONYMS = _merge_map(_BASE_SYNONYMS, _syn_raw, _MERGE_SYNONYMS)
ALIASES = _merge_map(_BASE_ALIASES, _alias_raw, _MERGE_ALIASES)


def _extract_page_id(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(?:pageid|page_id)\s*=\s*(\d+)", text, re.I)
    return m.group(1) if m else ""


_DOCKER_HINTS = [
    "docker", "container", "image", "dockerfile", "compose", "docker-compose",
    "registry", "volume", "bind", "mount", "run", "build", "ps", "stop",
    "start", "exec", "port", "network", "tag", "pull", "push",
    "??", "????", "???", "????", "?? ??", "??-???", "?? ???",
    "???", "?????", "??", "???", "??", "????", "??", "??",
    "??", "?????",
]

def _has_docker_intent(text: str) -> bool:
    s = (text or "").lower()
    return any(k.lower() in s for k in _DOCKER_HINTS)


def _docker_focus_query(text: str) -> str:
    base = (text or "").strip()
    extra = (
        "docker container image dockerfile docker-compose compose registry "
        "run build ps stop start exec volume mount network port "
        "tag pull push "
        "?? ???? ??? ???? ??-??? ?? ??? ??? "
        "????? ?? ??? ?? ???? ?? ?? ?? ?????"
    )
    return f"{base} {extra}".strip()

_REFUSAL_RE = re.compile(
    r"(정보|근거|컨텍스트|자료).*(부족|없|무관|부재|찾을 수 없음|없습니다)|"
    r"(답변할 수 없습니다|요약할 수 없습니다|제공할 수 없습니다)",
    re.I,
)
_REFUSAL_RE2 = re.compile(
    r"(insufficient|not enough|no).*(context|evidence)|cannot answer",
    re.I,
)

def _is_refusal_like(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return bool(_REFUSAL_RE.search(t) or _REFUSAL_RE2.search(t))

def _filter_items_by_keywords(items: list[dict] | None, keywords: list[str]) -> list[dict]:
    if not items:
        return []
    kws = [k.lower() for k in keywords if k]
    out = []
    for it in items:
        md = it.get("metadata") or {}
        blob = " ".join(
            [
                str(it.get("text") or ""),
                str(md.get("title") or ""),
                str(md.get("section_title") or ""),
            ]
        ).lower()
        if any(k in blob for k in kws):
            out.append(it)
    return out

# [PATCH] 쿼리 변형 제어용 플래그/상수 추가
USE_SYNONYMS = (os.getenv("ROUTER_USE_SYNONYMS", "0").lower() not in ("0","false","no"))
VARIANTS_MAX = int(os.getenv("ROUTER_VARIANTS_MAX", "4"))


_BASE_STOPWORDS = set(
    "\uC740 \uB294 \uC774 \uAC00 \uC744 \uB97C \uC5D0 \uC758 \uC640 \uACFC "
    "\uB3C4 \uB85C \uC73C\uB85C \uC5D0\uC11C \uC5D0\uAC8C \uADF8\uB9AC\uACE0 "
    "\uADF8\uB7EC\uB098 \uADF8\uB7EC\uC11C \uBB34\uC5C7 \uBB50\uC57C \uBB50\uC9C0 "
    "\uC124\uBA85 \uD574\uC918 \uB300\uD55C \uB300\uD574 \uC815\uB9AC "
    "\uAC1C\uC694 \uC18C\uAC1C \uC790\uC138\uD788"
    .split()
)
def _load_list_env(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [t for t in re.split(r"[,\s]+", raw) if t]

_STOPWORDS = set(_BASE_STOPWORDS)
_stop_env = set(_load_list_env("ROUTER_STOPWORDS"))
if _stop_env:
    _STOPWORDS = (_STOPWORDS | _stop_env) if _MERGE_STOPWORDS else set(_stop_env)

ROUTER_USE_FOCUS_KEYWORDS = os.getenv("ROUTER_USE_FOCUS_KEYWORDS", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)
_FOCUS_KEYWORDS: set[str] = set()
if ROUTER_USE_FOCUS_KEYWORDS:
    _focus_env = set(_load_list_env("ROUTER_FOCUS_KEYWORDS"))
    if _focus_env:
        _FOCUS_KEYWORDS = (
            (_FOCUS_KEYWORDS | _focus_env) if _MERGE_FOCUS_KEYWORDS else set(_focus_env)
        )

def _item_kind(it: dict) -> str:
    md = it.get("metadata") or {}
    return (md.get("kind") or it.get("kind") or "").strip()

def _ctx_ready_for_file(items: list[dict], ctx: str) -> bool:
    if not items:
        return False
    if len(ctx or "") >= ROUTER_QA_MIN_CTX_LEN:
        return True
    for it in items:
        kind = _item_kind(it)
        if kind and kind not in ("title", "summary"):
            return True
    return False

def _latest_upload_source() -> str:
    try:
        base = Path(ROUTER_UPLOADS_DIR)
    except Exception:
        return ""
    if not base.exists() or not base.is_dir():
        return ""
    exts = {".pdf", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".txt", ".md", ".log", ".docx"}
    files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return ""
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return f"/app/uploads/{latest.name}"

def _file_stem_for_query(src: str) -> str:
    name = Path(str(src)).name
    # Strip UUID prefix like "<uuid>_filename.ext"
    name = re.sub(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", "", name, flags=re.I)
    return Path(name).stem

def _match_upload_source_by_query(q: str) -> str:
    try:
        base = Path(ROUTER_UPLOADS_DIR)
    except Exception:
        return ""
    if not base.exists() or not base.is_dir():
        return ""
    q_norm = re.sub(r"\s+", "", _normalize_query(q).lower())
    q_tokens = [t for t in _tokens(q) if t not in _STOPWORDS]
    exts = {".pdf", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".txt", ".md", ".log", ".docx"}
    best = ("", 0)
    for p in base.iterdir():
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        stem = _file_stem_for_query(p.name).lower()
        stem_norm = re.sub(r"\s+", "", _normalize_query(stem))
        score = 0
        if q_norm and q_norm in stem_norm:
            score = max(score, 5)
        for t in q_tokens:
            if t and t in stem_norm:
                score += 1
        if score > best[1]:
            best = (p.name, score)
    if best[0] and best[1] > 0:
        return f"/app/uploads/{best[0]}"
    return ""

def _history_upload_source(messages: list[dict]) -> str:
    if not messages:
        return ""
    texts: list[str] = []
    for m in messages:
        if (m.get("role") or "") != "user":
            continue
        content = str(m.get("content") or "").strip()
        if content:
            texts.append(content)
    if not texts:
        return ""
    hint = " ".join(texts[-3:])
    return _match_upload_source_by_query(hint)

def _normalize_upload_candidate(value: str) -> str:
    if not value:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    if "/uploads/" in s:
        idx = s.lower().rfind("/uploads/")
        s = s[idx:] if idx >= 0 else s
        if not s.startswith("/app/"):
            if s.startswith("/uploads/"):
                s = "/app" + s
        return s
    exts = (".pdf", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".txt", ".md", ".log", ".docx")
    if s.lower().endswith(exts):
        matched = _match_upload_source_by_query(s)
        return matched or f"/app/uploads/{s}"
    return ""

def _is_probable_source(value: str) -> bool:
    if not value:
        return False
    s = str(value).strip()
    if not s:
        return False
    if _LOCAL_SRC_RE.search(s):
        return True
    if re.search(r"pageid\s*=\s*\d+", s, re.I):
        return True
    if s.lower().startswith("confluence:"):
        return True
    if s.lower().startswith(("http://", "https://")):
        return True
    if re.search(r"\.(pdf|pptx|ppt|xlsx|xls|csv|txt|md|log|docx)\b", s, re.I):
        return True
    return False

def _extract_sources_from_metadata(meta: dict) -> list[str]:
    if not isinstance(meta, dict):
        return []
    candidates: list[object] = []
    for k in ("source", "sources", "rag_source", "rag_sources", "file", "files", "attachments", "attachment"):
        v = meta.get(k)
        if v:
            candidates.append(v)
    flat: list[str] = []
    def _push(v: object) -> None:
        if not v:
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                _push(x)
        elif isinstance(v, dict):
            for k in ("rag_source", "source", "path", "name", "filename", "file_name", "url"):
                if v.get(k):
                    _push(v.get(k))
            for k in ("file", "data", "metadata"):
                if v.get(k):
                    _push(v.get(k))
        else:
            flat.append(str(v))
    for c in candidates:
        _push(c)

    local: list[str] = []
    other: list[str] = []
    for c in flat:
        norm = _normalize_upload_candidate(c)
        if norm:
            local.append(norm)
            continue
        if _is_probable_source(c):
            other.append(str(c).strip())
    ordered = local + other
    out: list[str] = []
    for s in ordered:
        if s and s not in out:
            out.append(s)
    return out

def _apply_src_filters(payload: dict, primary_src: str, src_set: list[str] | None) -> None:
    if src_set:
        payload["sources"] = src_set
    if primary_src:
        payload["source"] = primary_src

def _extract_upload_source_hint(raw_msgs: list[dict], rewrite_meta: dict) -> str:
    candidates: list[object] = []
    if isinstance(rewrite_meta, dict):
        ent = rewrite_meta.get("entities") or {}
        for k in ("SOURCE", "source", "file", "filename", "file_name", "attachment"):
            v = ent.get(k)
            if v:
                candidates.append(v)
        cons = rewrite_meta.get("constraints") or {}
        for k in ("SOURCE", "source", "file", "filename", "file_name", "attachment"):
            v = cons.get(k)
            if v:
                candidates.append(v)
    for m in raw_msgs or []:
        for k in ("attachments", "files", "file"):
            if k in m and m.get(k):
                candidates.append(m.get(k))
        content = m.get("content")
        if isinstance(content, dict):
            for k in ("attachments", "files", "file", "path", "name", "filename", "file_name"):
                v = content.get(k)
                if v:
                    candidates.append(v)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    for k in ("path", "name", "filename", "file_name", "file", "source", "url"):
                        v = part.get(k)
                        if v:
                            candidates.append(v)
    flat: list[str] = []
    def _push(v: object) -> None:
        if not v:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _push(x)
        elif isinstance(v, dict):
            for k in ("path", "name", "filename", "file_name", "url", "source"):
                if v.get(k):
                    _push(v.get(k))
        else:
            flat.append(str(v))
    for c in candidates:
        _push(c)
    for c in flat:
        s = _normalize_upload_candidate(c)
        if s:
            return s
    return ""

def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "") == "user":
            text = str(m.get("content") or "").strip()
            if text:
                return text
    return ""

def _token_set(text: str) -> set[str]:
    if not text:
        return set()
    toks = [t for t in _tokens(text) if t not in _STOPWORDS]
    return set(toks)

def _is_topic_shift(prev_text: str, curr_text: str) -> bool:
    prev = _token_set(prev_text)
    curr = _token_set(curr_text)
    if not prev or not curr:
        return False
    overlap = len(prev & curr) / max(1, min(len(prev), len(curr)))
    thresh = float(os.getenv("ROUTER_HISTORY_SIM_THRESH", "0.2") or 0.2)
    return overlap < thresh

def _is_followup_hint(text: str) -> bool:
    if not text:
        return False
    norm = _normalize_query(text)
    short_limit = int(os.getenv("ROUTER_HISTORY_SHORT_CHARS", "80") or 80)
    if len(norm) > short_limit:
        return False
    hints = [
        "더", "자세", "좀", "추가", "계속", "이어", "다시", "방금", "위에", "앞에",
        "그거", "그것", "이거", "이것", "이걸", "이것도",
    ]
    has_hint = any(h in norm for h in hints)
    return has_hint or len(norm) <= 12

def _is_explicit_reset(text: str) -> bool:
    if not text:
        return False
    norm = _normalize_query(text)
    resets = [
        "다른 문서", "새 문서", "새로운 문서", "문서 바꿔", "문서 변경", "주제 바꿔",
        "컨플루언스", "confluence", "pageid", "페이지",
    ]
    return any(r in norm for r in resets)

async def _resolve_source_from_index(client: httpx.AsyncClient, src: str) -> str:
    try:
        r = await client.get(f"{RAG}/index/sources")
        j = r.json() if hasattr(r, "json") else {}
        items = j.get("items") or []
    except Exception:
        return src
    if not items:
        return src
    stem = _file_stem_for_query(src)
    if not stem:
        return src
    candidates = []
    for it in items:
        s = str(it.get("source") or "")
        if not s:
            continue
        if _file_stem_for_query(s) == stem:
            candidates.append(s)
    if not candidates:
        return src
    candidates.sort(key=lambda v: len(v))
    return candidates[0]

async def _find_source_in_index(client: httpx.AsyncClient, src: str) -> str:
    try:
        r = await client.get(f"{RAG}/index/sources")
        j = r.json() if hasattr(r, "json") else {}
        items = j.get("items") or []
    except Exception:
        return ""
    if not items:
        return ""
    stem = _file_stem_for_query(src)
    if not stem:
        return ""
    for it in items:
        s = str(it.get("source") or "")
        if s and _file_stem_for_query(s) == stem:
            return s
    return ""

async def _wait_for_index_source(
    client: httpx.AsyncClient,
    src: str,
    wait_sec: float,
) -> str:
    deadline = time.monotonic() + max(0.0, wait_sec)
    while True:
        found = await _find_source_in_index(client, src)
        if found:
            return found
        if time.monotonic() >= deadline:
            return ""
        await asyncio.sleep(ROUTER_INGEST_WAIT_INTERVAL)

async def _query_source_with_terms(
    client: httpx.AsyncClient,
    src: str,
    terms: list[str],
    k: int = 10,
) -> tuple[list[dict], str]:
    for term in terms:
        if not term:
            continue
        payload = {"question": term, "k": k, "sticky": False, "need_fallback": False, "source": src}
        r = await client.post(f"{RAG}/query", json=payload)
        j = r.json() if hasattr(r, "json") else {}
        items = (j.get("items") or j.get("contexts") or [])
        if not items:
            continue
        ctx_list = (j.get("context_texts")
                    or [c.get("text","") for c in (j.get("contexts") or [])]
                    or [it.get("text","") for it in (j.get("items") or [])])
        ctx_list = extract_texts(items)
        ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:MAX_CTX_CHARS]
        if ctx:
            return items, ctx
    return [], ""

async def _query_with_wait(
    client: httpx.AsyncClient,
    payload: dict,
    file_hint: bool,
) -> dict:
    if not file_hint or ROUTER_INGEST_WAIT_SEC <= 0:
        return (await client.post(f"{RAG}/query", json=payload)).json()

    deadline = time.monotonic() + ROUTER_INGEST_WAIT_SEC
    last = {}
    while True:
        last = (await client.post(f"{RAG}/query", json=payload)).json()
        items = (last.get("items") or last.get("contexts") or [])
        ctx_list = (last.get("context_texts")
                    or [c.get("text","") for c in (last.get("contexts") or [])]
                    or [it.get("text","") for it in (last.get("items") or [])])
        if file_hint and items:
            ctx_list = extract_texts(items)
        ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:MAX_CTX_CHARS]
        if _ctx_ready_for_file(items, ctx):
            return last
        if time.monotonic() >= deadline:
            return last
        await asyncio.sleep(ROUTER_INGEST_WAIT_INTERVAL)

def _pick_max_tokens(req_max: Optional[int]) -> int:
    if req_max is None or req_max <= 0:
        return OUTPUT_TOKENS
    return req_max

def _dbg(msg: str) -> None:
    if ROUTER_DEBUG:
        print(f"[router] {msg}")

# [ADD] Clarification 헬퍼 함수들
def _is_number_choice(text: str) -> tuple[bool, int]:
    """사용자 입력이 숫자 선택(1, 2, 3 등)인지 확인"""
    text = text.strip()
    if re.match(r'^\d+$', text):
        try:
            num = int(text)
            if 1 <= num <= 10:  # 1~10번까지 허용
                return True, num
        except ValueError:
            pass
    return False, 0

def _save_clarification(session_id: str, candidates: list) -> None:
    """Clarification 후보 목록을 세션에 저장"""
    _CLARIFICATION_SESSIONS[session_id] = {
        "candidates": candidates,
        "timestamp": time.time()
    }
    # 오래된 세션 정리 (5분 이상)
    now = time.time()
    expired = [k for k, v in _CLARIFICATION_SESSIONS.items()
               if now - v["timestamp"] > _CLARIFICATION_TIMEOUT]
    for k in expired:
        del _CLARIFICATION_SESSIONS[k]

def _get_clarification_choice(session_id: str, choice_num: int) -> str | None:
    """세션에 저장된 후보 중 choice_num번째의 source 반환"""
    session = _CLARIFICATION_SESSIONS.get(session_id)
    if not session:
        return None
    candidates = session.get("candidates", [])
    if 1 <= choice_num <= len(candidates):
        return candidates[choice_num - 1].get("source")
    return None

def _format_clarification_message(candidates: list, message: str) -> str:
    """후보 목록을 마크다운 형식으로 포맷 (HTML 주석으로 SOURCE 숨김)"""
    lines = [message, ""]
    for cand in candidates:
        cand_id = cand.get("id", 0)
        title = cand.get("title", "제목 없음")
        doc_type = cand.get("type", "문서")
        source = cand.get("source", "")

        # 파일명에서 제목 추출 (제목이 없으면)
        if title == "제목 없음" and source:
            # /app/uploads/xxx_filename.pdf -> filename.pdf
            fname = source.split("/")[-1]
            # UUID 접두사 제거
            if "_" in fname and len(fname.split("_")[0]) > 30:
                fname = "_".join(fname.split("_")[1:])
            if fname:
                title = fname

        lines.append(f"**{cand_id}. {title}** ({doc_type})")
        # source를 HTML 주석으로 숨김 (다음 요청에서 파싱용)
        lines.append(f"<!--CLRF:{cand_id}|{source}-->")
        lines.append("")

    lines.append("---")
    lines.append("💡 **원하시는 문서 번호를 입력해주세요** (예: `1`, `2`, `3`)")

    return "\n".join(lines)


def _extract_clarification_source_from_history(prev_assistant: str, choice_num: int) -> str | None:
    """이전 assistant 메시지에서 선택한 번호에 해당하는 source 추출"""
    if not prev_assistant:
        return None

    # HTML 주석에서 CLRF:N|source 패턴 찾기
    pattern = rf'<!--CLRF:{choice_num}\|([^>]+)-->'
    match = re.search(pattern, prev_assistant)
    if match:
        return match.group(1)
    return None


def _is_clarification_response(text: str) -> bool:
    """assistant 메시지가 clarification 응답인지 확인"""
    # "문서 번호를 입력해주세요" 또는 "어떤 것을 원하시나요" 가 있으면 clarification 응답
    return bool(text and ("문서 번호를 입력해주세요" in text or "어떤 것을 원하시나요" in text))

def _attach_trace(resp: dict, trace_id: str) -> dict:
    if isinstance(resp, dict):
        resp.setdefault("trace_id", trace_id)
    return resp

def _add_trace(payload: dict, trace_id: str) -> dict:
    payload["trace_id"] = trace_id
    return payload

async def _rewrite_query(user_msg: str, raw_msgs: list[dict]) -> tuple[str, dict]:
    if not user_msg:
        return "", {}
    # 최근 N턴만 사용 (user/assistant 모두)
    turns = []
    for m in raw_msgs[-ROUTER_REWRITE_TURNS:]:
        role = (m.get("role") or "").strip()
        content = m.get("content") or ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        content = content.strip()
        if role and content:
            turns.append({"role": role, "content": content})
    sysmsg = {
        "role": "system",
        "content": (
            "당신은 검색용 질문 재작성기다. 대화 문맥을 참고해 "
            "사용자 질문을 독립형 질문으로 재작성하라. "
            "발전소 ID, 티켓번호, 시스템명, 스페이스 키, pageId 같은 고유값은 절대 삭제하지 마라. "
            "JSON으로만 답하라: {\"standalone_query\":\"...\",\"entities\":{},\"constraints\":{}}"
        )
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [sysmsg] + turns,
        "stream": False,
        "temperature": 0,
        "max_tokens": 256
    }
    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
        try:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            rj = r.json()
            raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
            data = json.loads(raw)
            sq = (data.get("standalone_query") or "").strip()
            return sq, {
                "entities": data.get("entities") or {},
                "constraints": data.get("constraints") or {},
            }
        except Exception:
            return "", {}

def _needs_korean_retry(text: str) -> bool:
    if not text:
        return False
    ko = len(_KO_CHAR_RE.findall(text))
    cjk = len(_CJK_CHAR_RE.findall(text))
    return cjk > 0

async def _ensure_korean(text: str, req: "ChatReq") -> str:
    if not ROUTER_FORCE_KOREAN or not _needs_korean_retry(text):
        return text
    sysmsg = {
        "role": "system",
        "content": (
            "당신은 한국어 원어민처럼 자연스럽게 말하는 인공지능입니다. "
            "지금부터 모든 대화는 한국어로만 진행합니다. "
            "아래 내용을 의미를 유지한 채 자연스러운 한국어로만 번역하세요."
        )
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [sysmsg, {"role": "user", "content": text}],
        "stream": False,
        "temperature": 0,
        "max_tokens": _clamp_max_tokens(sysmsg["content"], [{"role":"user","content":text}], req.max_tokens),
    }
    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
        try:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            rj = r.json()
            raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            cleaned = sanitize(clean_llm_output(raw)) or text
            return cleaned
        except (httpx.RequestError, ValueError):
            return text

async def _llm_direct_answer(limited_msgs: list[dict], req: "ChatReq") -> str:
    now_kst = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d (%a) %H:%M:%S %Z")
    sysmsg = {
        "role": "system",
        "content": (
            "당신은 한국어 원어민처럼 자연스럽게 말하는 인공지능입니다. "
            "지금부터 모든 대화는 한국어로만 진행합니다. "
            f"현재 날짜와 시간: {now_kst}. "
            "문서 인덱스가 없어도 일반 상식·수학·날짜/시간 등은 직접 답하세요. "
            "모든 답변은 반드시 한국어로만 작성해야 합니다. "
            "‘인덱스에 근거 없음’ 같은 말은 하지 마세요."
        )
    }
    max_tokens = _clamp_max_tokens(sysmsg["content"], limited_msgs, req.max_tokens)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [sysmsg] + limited_msgs,
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens
    }
    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
        try:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            rj = r.json()
            raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            if ROUTER_DEBUG and not raw:
                _dbg(f"llm_empty: status={r.status_code} keys={list(rj.keys())} err={rj.get('error')}")
            if not raw:
                r2 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                rj2 = r2.json()
                raw = rj2.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            content = sanitize(clean_llm_output(raw)) or "죄송해요. 지금은 답을 찾지 못했어요."
            if _needs_korean_retry(content):
                sysmsg["content"] = (
                    "당신은 한국어 원어민처럼 자연스럽게 말하는 인공지능입니다. "
                    "지금부터 모든 대화는 한국어로만 진행합니다. "
                    f"현재 날짜와 시간: {now_kst}. "
                    "모든 답변은 반드시 한국어로만 작성해야 합니다. "
                    "질문과 무관한 내용은 답하지 마세요."
                )
                payload["messages"] = [sysmsg] + limited_msgs
                r3 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                rj3 = r3.json()
                raw3 = rj3.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                content = sanitize(clean_llm_output(raw3)) or content
            content = await _ensure_korean(content, req)
            return content
        except (httpx.RequestError, ValueError):
            return "죄송해요. 지금은 답을 찾지 못했어요."

def _est_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token in mixed ko/en
    return max(1, math.ceil(len(text or "") / 4))

def _filter_urls_by_host(urls: list[str]) -> list[str]:
    if not urls:
        return []
    raw = os.getenv("ALLOWED_SOURCE_HOSTS", "").strip()
    if not raw:
        return urls
    allowed = {h.strip().lower() for h in raw.split(",") if h.strip()}
    if not allowed:
        return urls
    out = []
    for u in urls:
        try:
            host = (httpx.URL(u).host or "").lower()
        except Exception:
            continue
        if host in allowed:
            out.append(u)
    return out

def _clamp_max_tokens(system_prompt: str, messages: list[dict], req_max: Optional[int]) -> int:
    base = _pick_max_tokens(req_max)
    input_tokens = _est_tokens(system_prompt)
    for m in messages or []:
        input_tokens += _est_tokens(str(m.get("content") or ""))
    # leave safety margin for system + output
    remain = max(128, MODEL_LIMIT_TOKENS - input_tokens - SAFETY_MARGIN)
    return min(base, remain)

def _fit_ctx_to_budget(system_prompt: str, user_msg: str, ctx: str) -> str:
    if not ctx:
        return ""
    input_tokens = _est_tokens(system_prompt) + _est_tokens(user_msg)
    budget = MODEL_LIMIT_TOKENS - SAFETY_MARGIN - OUTPUT_TOKENS - input_tokens
    if budget <= 0:
        return ""
    max_chars = min(len(ctx), budget * 4)
    return ctx[:max_chars]

def _qa_score(j):
    if not j: return -1
    hits = float(j.get("hits") or 0)
    items = j.get("items") or []
    ctx = "\n\n".join(extract_texts(items))[:MAX_CTX_CHARS]
    return hits + (len(ctx) / 10000.0)  # 아주 단순 가중치

def _spaces_from_env():
    raw = os.getenv("CONFLUENCE_SPACE", "").strip()
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()] or None

SPACES = _spaces_from_env()

# [ADD] 질문에 가장 잘 맞는 space를 자동 선택 (없으면 None 반환 → 제한 없이 진행)
async def _auto_pick_spaces(q: str, client: httpx.AsyncClient) -> list[str] | None:
    if not SPACES or len(SPACES) <= 1:
        return SPACES  # 단일 스페이스거나 미설정이면 굳이 선택 안 함

    scores = []
    try:
        # 한 번만 탐색해도 충분하니 첫 변형 쿼리로 간단 프리뷰
        probe_q = q.strip()
        for sp in SPACES:
            # 프리뷰: LLM 생성 없이 검색만(k=3) 해서 '적합도' 점수 계산
            r = await client.post(f"{RAG}/query", json={ "q": probe_q, "k": 3, "sticky": False , "spaces": [sp] })

            j = r.json()
            # 컨텍스트 텍스트를 조금 모아서 질문과의 토큰 겹침(relevance_ratio)로 스코어링
            probe = j.get("items") or j.get("contexts") or j.get("documents") or j.get("chunks") or []
            ctx = "\n\n".join(extract_texts(probe))[:1500]
            score = float(j.get("hits") or 0) + relevance_ratio(probe_q, ctx)  # 간단 복합 점수
            scores.append((score, sp))
    except Exception as e:
        print(f"[router] space probe failed: {e}")

    # 최고 점수만 채택 (0 이하면 제한 하지 않음)
    if not scores:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[0]
    # 동률이거나 근소 차이면 제한하지 않고(None) 진행 → 오탐 방지
    if top[0] > 0 and (len(scores) == 1 or top[0] >= scores[1][0] + 0.2):
        return [top[1]]
    return None

_JOSA_PATTERN = os.getenv(
    "ROUTER_JOSA_PATTERN",
    r"(\uC73C\uB85C\uC368|\uC73C\uB85C\uC11C|\uC73C\uB85C\uC368|\uCC98\uB7FC|\uAE4C\uC9C0|"
    r"\uBD80\uD130|\uC5D0\uAC8C\uC11C|\uC5D0\uAC8C|\uD55C\uD14C|\uAED8\uC11C|\uB9CC\uD07C|"
    r"\uB3C4|\uACFC\uB294|\uC5D0\uC11C\uB294|\uC5D0\uC11C|\uC5D0\uAC8C|\uD55C\uD14C|"
    r"\uB4E4|\uACFC|\uB97C|\uB9CC|\uB300\uB85C)$",
)
_JOSA_RE = re.compile(_JOSA_PATTERN)

def _strip_josa(term: str) -> str:
    if not term:
        return ""
    return _JOSA_RE.sub("", term)

USE_KO_MORPH = (os.getenv("ROUTER_KO_MORPH", "0").lower() not in ("0","false","no"))
try:
    from kiwipiepy import Kiwi
    _KIWI_OK = True
except Exception:
    Kiwi = None
    _KIWI_OK = False

@lru_cache
def _get_kiwi():
    return Kiwi() if _KIWI_OK else None

def _tokens(s: str) -> list[str]:
    if USE_KO_MORPH and _KIWI_OK:
        kiwi = _get_kiwi()
        toks = []
        for t in kiwi.tokenize(s or ""):
            pos = getattr(t, "pos", None) or getattr(t, "tag", "")
            if pos and str(pos).startswith("J"):
                continue
            form = (t.form or "").strip().lower()
            if form:
                toks.append(form)
        return toks
    toks = []
    for t in _KO_EN_TOKEN.findall(s or ""):
        t2 = _strip_josa(t).lower()
        if t2:
            toks.append(t2)
    return toks

def _core_query_terms(query: str, max_terms: int = 4) -> list[str]:
    toks = [t for t in _tokens(query) if t not in _STOPWORDS]
    if not toks:
        return []
    filtered = [t for t in toks if len(t) >= 2 or re.search(r"\d", t)]
    if not filtered:
        filtered = toks
    idx = {}
    for i, t in enumerate(filtered):
        if t not in idx:
            idx[t] = i
    ranked = sorted(idx.items(), key=lambda kv: (-len(kv[0]), kv[1]))
    return [t for t, _ in ranked[:max_terms]]

def relevance_ratio(q: str, ctx: str, ctx_limit: int = 2000) -> float:
    qk = [t for t in _tokens(q) if t not in _STOPWORDS]
    if not qk:
        return 0.0

    # [추가] 공백 제거 후 부분문자열 매칭(한글 합성어 대응)
    qnorm = re.sub(r'\s+', '', _normalize_query(q).lower())
    cnorm = re.sub(r'\s+', '', (ctx or "")[:ctx_limit].lower())
    if len(qnorm) >= 4 and qnorm in cnorm:
        return 1.0

    ck = set(_tokens((ctx or "")[:ctx_limit]))
    common = sum(1 for t in qk if t in ck)
    return common / len(qk)


# 환경변수로 문턱값 조정 가능 (기본 0.2)
REL_THRESH = float(os.getenv("ROUTER_MIN_OVERLAP", "0.06"))


def supported_by_context(answer: str, ctx: str) -> bool:
    if not answer or not ctx:
        return False
    ov = relevance_ratio(answer, ctx, ctx_limit=MAX_CTX_CHARS)
    return ov >= ANSWER_MIN_OVERLAP


def _normalize_query(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_relevant(q: str, ctx: str) -> bool:
    return relevance_ratio(q, ctx) >= REL_THRESH

def _is_webui_task(s: str) -> bool:
    return bool(re.match(r"(?is)^\s*#{3}\s*task\s*:", (s or "")))

def _last_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "") == "assistant":
            content = str(m.get("content") or "").strip()
            if content:
                return content
    return ""

def _assistant_used_rag(text: str) -> bool:
    if not text:
        return False
    return ("RAG" in text) and (("근거:" in text) or ("Evidence:" in text))

def _source_query_overlap(src: str, query: str) -> bool:
    if not src or not query:
        return False
    stem_terms = [t for t in _tokens(_file_stem_for_query(src)) if t not in _STOPWORDS]
    q_terms = [t for t in _tokens(query) if t not in _STOPWORDS]
    if not stem_terms or not q_terms:
        return False
    return any(t in stem_terms for t in q_terms)

def _focus_query_for_source(query: str) -> str:
    if not query:
        return ""
    core = _core_query_terms(query, max_terms=3)
    if core:
        return " ".join(core)
    if not ROUTER_USE_FOCUS_KEYWORDS or not _FOCUS_KEYWORDS:
        return ""
    q_lower = query.lower()
    toks = [t for t in _tokens(query) if t not in _STOPWORDS]
    for key in _FOCUS_KEYWORDS:
        if key in toks or key in q_lower:
            return key
    uniq = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= 3:
            break
    return " ".join(uniq) if uniq else query

def _allow_llm_fallback(
    user_msg: str,
    file_hint: bool,
    spaces_hint: list[str] | None,
    rag_followup: bool,
    topic_shift: bool,
) -> bool:
    if not ALLOW_LLM_FALLBACK:
        return False
    if rag_followup:
        return False
    if topic_shift:
        return True
    if file_hint:
        return False
    if _CONF_HOST_RE.search(user_msg or "") or _CONF_HINT_RE.search(user_msg or ""):
        return False
    if _PAGEID_HINT_RE.search(user_msg or ""):
        return False
    if spaces_hint and _FILE_HINT_RE.search(user_msg or ""):
        return False
    return True

def _route_state(
    user_msg: str,
    file_hint: bool,
    spaces_hint: list[str] | None,
    rag_followup: bool,
    topic_shift: bool,
) -> tuple[str, str]:
    if file_hint or spaces_hint or _CONF_HOST_RE.search(user_msg or "") or _CONF_HINT_RE.search(user_msg or ""):
        return "RAG_REQUIRED", "file_or_space_or_confluence"
    if _PAGEID_HINT_RE.search(user_msg or ""):
        return "RAG_REQUIRED", "pageid"
    if ROUTER_USE_HINTS and _RAG_REQUIRED_HINTS and _RAG_REQUIRED_HINTS.search(user_msg or ""):
        return "RAG_REQUIRED", "required_hints"
    if rag_followup and not topic_shift:
        return "RAG_PREFERRED", "rag_followup"
    if ROUTER_USE_HINTS and _RAG_PREFERRED_HINTS and _RAG_PREFERRED_HINTS.search(user_msg or ""):
        return "RAG_PREFERRED", "preferred_hints"
    if topic_shift:
        return "NO_RAG", "topic_shift"
    return "NO_RAG", "default"

async def _llm_route_state(
    user_msg: str,
    rag_followup: bool,
    topic_shift: bool,
) -> tuple[str | None, str]:
    if not user_msg:
        return None, "llm_empty"
    if ROUTER_LLM_ROUTE_SCORE:
        prompt = (
            "You are a routing scorer. Return a single integer 0-10.\n"
            "0 = NO_RAG (chit-chat, common knowledge, general dev knowledge).\n"
            "10 = RAG_REQUIRED (needs internal docs: policy, operations, logs, contracts, prices, "
            "confluence pages, internal files).\n"
            "Scores 7-8 mean RAG_PREFERRED (docs improve quality).\n"
            "Output only the number.\n"
        )
    else:
        prompt = (
            "You are a router that must output exactly one label.\n"
            "Labels: RAG_REQUIRED, RAG_PREFERRED, NO_RAG.\n"
            "RAG_REQUIRED: internal docs required (company policy, operations, logs, contracts, prices).\n"
            "RAG_PREFERRED: general knowledge possible but docs improve quality.\n"
            "NO_RAG: chit-chat, common knowledge, general dev knowledge.\n"
            "Output only the label.\n"
        )
    user = (
        f"question: {user_msg}\n"
        f"rag_followup: {rag_followup}\n"
        f"topic_shift: {topic_shift}\n"
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 5,
    }
    try:
        timeout = httpx.Timeout(ROUTER_LLM_ROUTE_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        if ROUTER_LLM_ROUTE_SCORE:
            match = re.search(r"\b(10|[0-9])\b", text)
            if not match:
                return None, "llm_invalid"
            score = int(match.group(1))
            if score >= ROUTER_LLM_ROUTE_REQUIRED_THRESHOLD:
                return "RAG_REQUIRED", "llm_score"
            if score >= ROUTER_LLM_ROUTE_SCORE_THRESHOLD:
                return "RAG_PREFERRED", "llm_score"
            return "NO_RAG", "llm_score"
        label = text.strip().upper()
        if label in {"RAG_REQUIRED", "RAG_PREFERRED", "NO_RAG"}:
            return label, "llm_route"
        return None, "llm_invalid"
    except Exception:
        return None, "llm_error"


def _has_local_items(items: list[dict]) -> bool:
    for it in items or []:
        src = (it.get("metadata") or {}).get("source") or it.get("source") or ""
        if src and _LOCAL_SRC_RE.search(str(src)):
            return True
    return False


async def _precheck_rag_local(user_msgs: list[str]) -> bool:
    timeout = httpx.Timeout(
        connect=ROUTER_PRECHECK_CONNECT_TIMEOUT,
        read=ROUTER_PRECHECK_TIMEOUT,
        write=ROUTER_PRECHECK_TIMEOUT,
        pool=ROUTER_PRECHECK_CONNECT_TIMEOUT,
    )
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for msg in user_msgs:
            if not msg or not msg.strip():
                continue
            payload = {"question": msg, "k": ROUTER_PRECHECK_K, "need_fallback": False}
            for attempt in range(max(1, ROUTER_PRECHECK_RETRIES + 1)):
                try:
                    resp = await client.post(f"{RAG}/query", json=payload)
                    try:
                        data = resp.json()
                    except Exception as exc:
                        if ROUTER_DEBUG:
                            _dbg(
                                "precheck_rag_error: q='%s' err=%s status=%s"
                                % (msg[:60], f"{exc.__class__.__name__}: {exc}", resp.status_code)
                            )
                        data = None
                    break
                except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                    if ROUTER_DEBUG:
                        _dbg(
                            "precheck_rag_error: q='%s' err=%s attempt=%s"
                            % (msg[:60], f"{exc.__class__.__name__}: {exc}", attempt + 1)
                        )
                    if attempt < ROUTER_PRECHECK_RETRIES:
                        await asyncio.sleep(0.2)
                    data = None
                except Exception as exc:
                    if ROUTER_DEBUG:
                        _dbg(f"precheck_rag_error: q='{msg[:60]}' err={exc.__class__.__name__}: {exc}")
                    data = None
                    break
            if not data:
                continue
            items = data.get("items") or []
            local_ok = _has_local_items(items)
            if ROUTER_DEBUG:
                _dbg(f"precheck_rag: q='{msg[:60]}' hits={data.get('hits')} items={len(items)} local={local_ok}")
            if local_ok:
                return True
    return False

app = FastAPI()

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List[Msg]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    metadata: Optional[dict] = None

def strip_reasoning(text: str) -> str:
    if not text:
        return text
    # assistant_thought 블록 제거 (response 토큰 전까지)
    text = re.sub(r'(?is)<\|assistant_thought\|>.*?(?=<\|assistant_response\|>|\Z)', '', text)
    # response 토큰 마커 제거
    text = re.sub(r'(?is)<\|assistant_response\|>', '', text)

    # <think>…</think> 또는 </think> 없이 끝까지
    text = re.sub(r'(?is)<think\b[^>]*>.*?(?:</think>|$)', '', text)

    # 코드블록 형태의 생각/추론
    text = re.sub(r'(?is)```(?:thinking|reasoning|thought|scratchpad)[\s\S]*?```', '', text)

    # “Thought:” “Reasoning:” 스타일(빈 줄까지)
    text = re.sub(r'(?im)^\s*(?:thought|reasoning|scratchpad)\s*:\s*[\s\S]*?(?:\n{2,}|\Z)', '', text)

    return text.strip()

def clean_llm_output(text: str) -> str:
    cleaned = strip_reasoning(text or "").strip()
    if cleaned:
        return cleaned
    # If the model put the answer inside <think>, recover it.
    m = re.search(r'(?is)<think\b[^>]*>(.*?)</think>', text or "")
    if m:
        recovered = (m.group(1) or "").strip()
        if recovered:
            return recovered
    # Fallback: if <think> is unclosed, keep everything after the tag.
    if re.search(r'(?is)<think\b[^>]*>', text or ""):
        return re.sub(r'(?is)<think\b[^>]*>', '', text or "").strip()
    # Last resort: drop <think> tags only.
    return re.sub(r'(?is)</?think[^>]*>', '', text or "").strip()

def build_final_only_prompt(ctx_text: str) -> str:
    return (
        "You must answer only with the final answer. Do not include <think> or reasoning.\n"
        "Answer strictly in Korean.\n"
        "If you cannot answer from the context, reply exactly: 인덱스에 근거 없음\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def build_force_context_prompt(ctx_text: str) -> str:
    return (
        "You must answer using only the provided context.\n"
        "Answer strictly in Korean.\n"
        "Even if the context is partial, provide the best-effort summary.\n"
        "Do NOT say '인덱스에 근거 없음' or refuse.\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def _httpx_timeout() -> httpx.Timeout:
    connect = float(os.getenv("ROUTER_HTTP_CONNECT_TIMEOUT", "5"))
    read = float(os.getenv("ROUTER_HTTP_READ_TIMEOUT", "60"))
    write = float(os.getenv("ROUTER_HTTP_WRITE_TIMEOUT", "60"))
    pool = float(os.getenv("ROUTER_HTTP_POOL_TIMEOUT", "5"))
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)

def _fallback_summary_from_ctx(ctx_text: str, max_chars: int = 1200) -> str:
    # Strip source tags and keep a short, readable excerpt as last resort.
    text = re.sub(r"\[SOURCE:[^\]]+\]\s*", "", ctx_text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

def _msg_to_dict(m) -> dict:
    if isinstance(m, dict):
        return {"role": m.get("role", "user"), "content": m.get("content", "")}
    if hasattr(m, "dict"):
        d = m.dict()
        return {"role": d.get("role", "user"), "content": d.get("content", "")}
    return {"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))}

async def _limited_messages(messages) -> list[dict]:
    msgs = [_msg_to_dict(m) for m in (messages or []) if _msg_to_dict(m).get("content") is not None]
    if not msgs:
        return []
    system = [m for m in msgs if m.get("role") == "system"]
    others = [m for m in msgs if m.get("role") != "system"]
    if not others:
        return system[:1] if system else []
    budget = max(0, ROUTER_USER_MAX_CHARS)
    total = 0
    keep = []
    for m in reversed(others):
        content = m.get("content") or ""
        total += len(content)
        keep.append(m)
        if budget and total >= budget:
            break
    keep.reverse()
    if system:
        return [system[0]] + keep
    return keep

def _replace_last_user(messages: list[dict], content: str) -> list[dict]:
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {"role": "user", "content": content}
            break
    return out


def mark_lonely_numbers_as_total(text: str) -> str:
    """
    줄 전체가 숫자만으로 이루어진 경우 '(합계: N)'으로 바꿔
    LLM이 개별 항목 수치로 오해하지 않도록 힌트를 준다.
    """
    def repl(m: re.Match):
        n = m.group(1)
        return f"(합계: {n})"
    return _NUM_ONLY_LINE.sub(repl, text)

# [추가] 컨텍스트가 '목록스러움'을 보이는지 가볍게 스코어링
def _looks_structured(ctx: str) -> bool:
    if not ctx: return False
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    bullet_like = 0
    for ln in lines[:40]:
        if re.match(r"^(?:[-•*]\s+|\d+\.\s+|\[\w+\]\s+)", ln):
            bullet_like += 1
        elif len(ln) <= 28:
            bullet_like += 0.5
    return bullet_like >= 4

def pick_answer_mode(user_msg: str, ctx_text: str) -> str:
    if ANSWER_MODE != "auto":
        return ANSWER_MODE
    um = (user_msg or "").lower()
    if any(k.lower() in um for k in _BULLET_HINTS):
        return "bulleted"
    if any(k.lower() in um for k in _PARA_HINTS):
        return "paragraph"
    return "bulleted" if _looks_structured(ctx_text) else "paragraph"


# --- utils ----------------------------------------------------

def normalize_query_router(q: str) -> str:
    if not q: return ""
    s = q.strip()
    s = _CONTEXT_CUT_RE.sub("", s)
    if ("---" in s or "—" in s or "–" in s) and re.search(r"\[?컨텍스트\]|\[?context\]", s, re.I):
        m = re.search(r"\[?컨텍스트\]|\[?context\]", s, re.I)
        if m:
            s = s[:m.start()].rstrip()
    s = re.sub(r'(?i)\bstfp\b|\bsfttp\b|\bsfpt\b|\bsftp\b', 'SFTP', s)
    s = s.replace("스텝", "SFTP")
    return s

def _expand_synonyms(s: str) -> list[str]:
    out = [s]
    for k, vs in SYNONYMS.items():
        if k in s:
            for v in vs:
                out.append(s.replace(k, v))
    ss = s.replace("상가 정보", "상가정보")
    if ss != s: out.append(ss)
    return list(dict.fromkeys(out))

# [PATCH] 변형 개수 제한 + 동의어 확장 ON/OFF
def generate_query_variants(q: str, limit: int | None = None) -> List[str]:
    limit = limit or VARIANTS_MAX
    s = normalize_query_router(q)
    cand: list[str] = []

    def add(x: str):
        x = re.sub(r'\s+', ' ', x).strip()
        if x and x not in cand:
            cand.append(x)

    # 필수 최소 변형만
    add(s)
    add(re.sub(r'\s+', '', s))  # 공백 제거
    add(re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', s))  # KR-EN 경계
    add(re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', s))

    # 동의어 확장은 환경변수로 끄고 켤 수 있게
    if USE_SYNONYMS:
        for v in _expand_synonyms(s):
            add(v)
            add(re.sub(r'\s+', '', v))

    core_terms = _core_query_terms(s, max_terms=4)
    if core_terms:
        add(" ".join(core_terms))
        add(re.sub(r'\s+', '', " ".join(core_terms)))

    return cand[:limit]


def sanitize(text: str) -> str:
    if not text: return ""
    t = unescape(text.replace("&nbsp;", " "))
    t = re.sub(r'(?i)(password|passwd|pwd|pass|pw|패스워드|비밀번호|비번|루트\s*비밀번호|root\s*password)\s*[:=]\s*\S+', r'\1: ******', t)
    t = re.sub(r'(?i)(token|secret|key|키)\s*[:=]\s*\S{6,}', r'\1: <redacted>', t)
    t = re.sub(r'(?i)(account|user(?:name)?|userid|id|계정|아이디)\s*[:=]\s*\S+', r'\1: <redacted>', t)
    t = re.sub(r'(?i)\b([a-z0-9._%+-]+)\s*/\s*(\S{4,})\b', r'\1 / ******', t)
    t = re.sub(r'(?i)\b(user|root)\s*/\s*(\S{4,})\b', r'\1 / ******', t)
    t = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b', r'\1.xxx', t)
    return t

def build_system_with_context(ctx_text: str, mode: str, force_summary: bool = False) -> str:
    if mode == "bulleted":
        style = (
            f"- 최대 {BULLETS_MAX}개 불릿으로 **구체적**으로 서술한다.\n"
            "- 각 불릿은 2~4문장으로 쓴다.\n"
            "- 불릿 외의 군더더기 서론/결론 문단은 길게 넣지 않는다.\n"
        )
    elif mode == "sections":
        style = (
            "- 2~4개의 **문단**으로 핵심→배경→세부→시사점 순으로 정리한다.\n"
            "- 마크다운 리스트 문법은 사용하지 않는다.\n"
        )
    else:
        style = (
            "- **리스트/번호/하이픈(-, •, 1.) 없이** 한두 개의 **연속된 문단**으로 자연스럽게 작성한다.\n"
            "- 첫 문장에 요지를 분명히 말하고, 이어서 구성요소·동작·제약을 설명한다.\n"
        )

    numeric_rules = (
        "- 표/목록의 **수치**는 **같은 행(같은 항목)** 에 적힌 숫자만 인용한다.\n"
        "- **합계/총계** 숫자를 개별 항목 값으로 배정하지 않는다.\n"
        "- 숫자를 쓸 때는 반드시 `항목명 숫자`로 **쌍**을 이뤄 서술한다. (예: `반포동 47`)\n"
        "- 상위 단위 합계는 필요 시 `(서초구 합계 439)`처럼 **합계임을 명시**한다.\n"
        "- 불명확하면 숫자 대신 '수치 불분명'으로 적는다.\n"
    )
    heading_hint = (f"- 가능하면 '{HEADING}' 아래로 정리한다.\n" if HEADING else "")

    if force_summary:
        guard_rule = (
            "- 질문과 컨텍스트가 완전히 일치하지 않더라도 컨텍스트 내용을 요약한다.\n"
            "- `인덱스에 근거 없음`은 출력하지 않는다.\n"
        )
    else:
        guard_rule = (
            "- 컨텍스트가 완전히 비었으면 정확히 `인덱스에 근거 없음`만 출력한다.\n"
            "- 컨텍스트가 있으면 '없음/불가/부족/모르' 류 표현을 피하고 요약한다.\n"
        )

    return (
        "역할: 주어진 컨텍스트를 근거로 **정확하고 실무 친화적인** 한국어 답변을 작성한다.\n"
        "원칙:\n"
        "- 반드시 한국어로만 작성한다.\n"
        "- 컨텍스트에 있는 정보만 사용하고 추측 금지.\n"
        "- 고유명사/수치는 가능한 그대로 인용하되 과도한 반복은 피한다.\n"
        "- 내부 추론(<think> 등) 출력 금지, 최종 답만 출력한다.\n"
        "- <think> 태그를 사용하더라도 답변은 반드시 태그 밖에 출력한다.\n"
        + heading_hint + style + numeric_rules +
        guard_rule +
        "- 민감정보(비밀번호/토큰/IP 마지막 옥텟)는 마스킹한다.\n"
        "[컨텍스트 시작]\n"
        f"{ctx_text}\n"
        "[컨텍스트 끝]\n"
    )

def _build_system_prefix(mode: str) -> str:
    # Use empty context to approximate system prompt size for budgeting.
    return build_system_with_context("", mode)

def _limit_urls(urls: List[str] | None, top_n: int = ROUTER_SOURCES_MAX) -> List[str]:
    out, seen = [], set()
    for u in urls or []:
        nu = _normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu); out.append(nu)
        if len(out) >= top_n:
            break
    return out

def _items_have_local_source(items: list[dict] | None) -> bool:
    for it in items or []:
        meta = it.get("metadata") or {}
        src = it.get("source") or meta.get("source") or ""
        if _LOCAL_SRC_RE.search(str(src)):
            return True
    return False

def _get_item_source(it: dict) -> str:
    meta = it.get("metadata") or {}
    src = it.get("source") or meta.get("source") or ""
    return str(src or "")

def _source_match_score(src: str, query: str) -> int:
    if not src or not query:
        return 0
    base = os.path.basename(src)
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    tokens = re.findall(r"[0-9A-Za-z]+|[가-힣]+", base)
    q = query or ""
    score = 0
    for t in tokens:
        if t and t in q:
            score += len(t)
    return score

def _prefer_single_source(items: list[dict], query: str = "") -> tuple[list[dict], str]:
    counts: dict[str, int] = {}
    scores: dict[str, int] = {}
    for it in items or []:
        src = _get_item_source(it)
        if not src:
            continue
        counts[src] = counts.get(src, 0) + 1
        if query:
            scores[src] = max(scores.get(src, 0), _source_match_score(src, query))
    if not counts:
        return items, ""
    # If any source matches the query, prefer the best match.
    if scores and max(scores.values()) > 0:
        primary = max(scores, key=lambda s: (scores[s], counts.get(s, 0)))
    else:
        local = [s for s in counts if _LOCAL_SRC_RE.search(s)]
        primary = max(local, key=lambda s: counts[s]) if local else max(counts, key=counts.get)
    filtered = [it for it in items or [] if _get_item_source(it) == primary]
    return filtered, primary

def _label_for_context(items: list[dict] | None, urls: list[str] | None) -> str:
    if not items and not urls:
        return "없음(LLM)"
    if _items_have_local_source(items):
        return "로컬 문서(RAG)"
    for u in urls or []:
        if _CONF_HOST_RE.search(u or ""):
            return "Confluence(RAG)"
    return "RAG"


def extract_texts(items: List[dict]) -> List[str]:
    texts = []
    for it in items or []:
        for key in ("text","content","chunk","snippet","body","page_text"):
            val = it.get(key)
            if isinstance(val,str) and val.strip():
                texts.append(unescape(val.strip())); break
        else:
            payload = it.get("payload") or it.get("data") or {}
            if isinstance(payload,dict):
                for key in ("text","content","body"):
                    val = payload.get(key)
                    if isinstance(val,str) and val.strip():
                        texts.append(unescape(val.strip())); break
    return texts

# [추가] URL 정규화(Confluence pageId 기준으로 중복 제거)
def _normalize_url(u: str) -> str:
    if not u:
        return ""
    u = str(u).split("#")[0].strip().rstrip("/")
    m = re.search(r"(pageId=\d+)", u)
    if m:
        base = u.split("?")[0]
        return f"{base}?{m.group(1)}"
    return u

def _collect_urls_from_items(items: List[dict], top_n: Optional[int] = None) -> List[str]:
    top_n = top_n or ROUTER_SOURCES_MAX
    cands = []

    def push(it: dict):
        if not isinstance(it, dict):
            return
        score = float(it.get("score") or it.get("similarity") or 0.0)
        url = it.get("url") or it.get("source_url") or it.get("link")
        if url:
            cands.append((score, _normalize_url(str(url))))
        payload = it.get("payload") or it.get("data") or {}
        if isinstance(payload, dict):
            url2 = payload.get("url") or payload.get("source_url") or payload.get("link")
            if url2:
                cands.append((score, _normalize_url(str(url2))))
        meta = it.get("metadata") or {}
        if isinstance(meta, dict):
            url3 = meta.get("url")
            if url3:
                cands.append((score, _normalize_url(str(url3))))
            if not (url or (payload if isinstance(payload, dict) else {}).get("url") or url3):
                src = meta.get("source")
                if src:
                    cands.append((score, str(src)))

    for it in items or []:
        push(it)

    cands = [(s, u) for (s, u) in cands if u]
    cands.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    for _, u in cands:
        if u not in out:
            out.append(u)
        if len(out) >= top_n:
            break
    return out

def is_good_context_for_qa(ctx: str) -> bool:
    if not ctx or not ctx.strip(): return False
    if len(ctx) < 180: return False
    if ctx.count("\n") < 2: return False
    return True

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": ROUTER_MODEL_ID, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    trace_id = str(uuid.uuid4())
    orig_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "").strip()
    clean_user_msg = normalize_query_router(orig_user_msg)
    page_id = _extract_page_id(orig_user_msg) or _extract_page_id(clean_user_msg)
    confluence_hint = bool(
        _CONF_HOST_RE.search(orig_user_msg or "")
        or _CONF_HOST_RE.search(clean_user_msg or "")
        or _CONF_HINT_RE.search(orig_user_msg or "")
        or _CONF_HINT_RE.search(clean_user_msg or "")
        or _PAGEID_HINT_RE.search(orig_user_msg or "")
        or _PAGEID_HINT_RE.search(clean_user_msg or "")
    )
    force_mcp = bool(page_id or confluence_hint)
    metadata = req.metadata or {}
    meta_sources = _extract_sources_from_metadata(metadata)
    meta_primary_src = meta_sources[0] if meta_sources else ""
    meta_src_set = meta_sources if len(meta_sources) > 1 else None
    file_hint = bool(_FILE_HINT_RE.search(orig_user_msg))
    if meta_sources and not file_hint:
        file_hint = True
    if force_mcp:
        file_hint = False
        meta_sources = []
        meta_primary_src = ""
        meta_src_set = None
    # limited_msgs can drop history; use raw req messages for source inference and rewrite.
    raw_msgs = [m.dict() if hasattr(m, "dict") else m for m in req.messages]
    rewrite_q = ""
    rewrite_meta = {}
    if ROUTER_REWRITE and not _is_webui_task(orig_user_msg):
        rewrite_q, rewrite_meta = await _rewrite_query(clean_user_msg, raw_msgs)
        if rewrite_q and ROUTER_DEBUG:
            _dbg(f"rewrite: trace_id={trace_id} standalone='{rewrite_q}' meta={rewrite_meta}")
    variants = generate_query_variants(clean_user_msg)
    if rewrite_q and rewrite_q != clean_user_msg:
        for v in generate_query_variants(rewrite_q):
            if v not in variants:
                variants.append(v)
    if force_mcp:
        variants = [rewrite_q or clean_user_msg] if (rewrite_q or clean_user_msg) else []
    limited_msgs = await _limited_messages(req.messages)
    limited_msgs = _replace_last_user(limited_msgs, clean_user_msg)
    prev_assistant_msg = _last_assistant_text(raw_msgs[:-1])
    prev_user_msg = _last_user_text(raw_msgs[:-1])
    topic_shift = bool(prev_user_msg) and _is_topic_shift(prev_user_msg, clean_user_msg)
    rag_followup = bool(prev_assistant_msg) and _assistant_used_rag(prev_assistant_msg) \
        and len(clean_user_msg) <= FOLLOWUP_MAX_CHARS and not topic_shift
    spaces_hint: list[str] | None = None
    history_src = _history_upload_source(raw_msgs[:-1]) if not force_mcp else ""
    file_hint_src = _extract_upload_source_hint(raw_msgs, rewrite_meta) if (file_hint and not force_mcp) else ""
    if meta_primary_src:
        file_hint_src = meta_primary_src
        if not history_src:
            history_src = meta_primary_src
    if file_hint and not file_hint_src:
        fallback_src = history_src or _latest_upload_source()
        if fallback_src:
            file_hint_src = fallback_src
            if not history_src:
                history_src = fallback_src
    if file_hint_src and not history_src:
        history_src = file_hint_src
    if meta_sources:
        _dbg(f"meta_sources: srcs={meta_sources}")
    if file_hint_src:
        _dbg(f"file_hint_src: src='{file_hint_src}'")

    # [ADD] Clarification 번호 선택 처리
    # 사용자가 숫자만 입력하면 이전 질문으로 다시 clarification 요청해서 source 가져옴
    clarification_choice_src = None
    is_choice, choice_num = _is_number_choice(orig_user_msg)
    _dbg(f"clrf_check: is_choice={is_choice} num={choice_num} prev_user={bool(prev_user_msg)} prev_asst={bool(prev_assistant_msg)}")
    if prev_assistant_msg:
        is_clrf = _is_clarification_response(prev_assistant_msg)
        _dbg(f"clrf_check: is_clrf_resp={is_clrf} asst_preview='{prev_assistant_msg[:80]}'")
    if is_choice and prev_user_msg and prev_assistant_msg and _is_clarification_response(prev_assistant_msg):
        _dbg(f"clarification_choice_attempt: choice={choice_num} prev_q='{prev_user_msg[:50]}'")
        # 이전 질문으로 다시 clarification 요청해서 candidates 가져오기
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as cl:
                re_payload = {"q": prev_user_msg, "k": 5, "sticky": False}
                re_resp = (await cl.post(f"{RAG}/query", json=re_payload)).json()
                if re_resp.get("clarification_needed"):
                    candidates = re_resp.get("candidates", [])
                    if 1 <= choice_num <= len(candidates):
                        clarification_choice_src = candidates[choice_num - 1].get("source")
                        _dbg(f"clarification_choice_resolved: choice={choice_num} src='{clarification_choice_src}'")
        except Exception as e:
            _dbg(f"clarification_choice_error: {e}")

        if clarification_choice_src:
            # 선택한 source를 file_hint_src로 사용
            file_hint_src = clarification_choice_src
            file_hint = True
            history_src = clarification_choice_src
        else:
            _dbg(f"clarification_choice: invalid choice={choice_num}")
    inferred_src = ""
    if not force_mcp:
        inferred_src = _match_upload_source_by_query(clean_user_msg)
        if inferred_src and inferred_src != file_hint_src and _source_query_overlap(inferred_src, clean_user_msg):
            file_hint = True
            file_hint_src = inferred_src
            if history_src != inferred_src:
                history_src = inferred_src
            _dbg(f"forced_infer_src: src='{inferred_src}' reason=query_match")
    if history_src and not force_mcp and not _source_query_overlap(history_src, clean_user_msg):
        topic_shift = True
    if history_src and not force_mcp and prev_user_msg and _is_topic_shift(prev_user_msg, clean_user_msg):
        if _is_explicit_reset(clean_user_msg):
            _dbg(
                "history_source_reset: src='%s' prev='%s' curr='%s' reason=explicit_reset"
                % (history_src, prev_user_msg[:80], clean_user_msg[:80])
            )
            history_src = ""
        elif _is_followup_hint(clean_user_msg):
            _dbg(
                "history_source_keep: src='%s' prev='%s' curr='%s' reason=followup"
                % (history_src, prev_user_msg[:80], clean_user_msg[:80])
            )
        else:
            _dbg(
                "history_source_reset: src='%s' prev='%s' curr='%s' reason=topic_shift"
                % (history_src, prev_user_msg[:80], clean_user_msg[:80])
            )
            history_src = ""
    if history_src:
        _dbg(f"history_source: src='{history_src}'")
    _dbg(f"req: trace_id={trace_id} user_len={len(orig_user_msg)} file_hint={file_hint} stream={bool(req.stream)} max_tokens={req.max_tokens}")

    # 메타 태스크면 RAG 건너뛰고 그대로 모델로 전달 (JSON 형식 보존)
    if _is_webui_task(orig_user_msg):
        max_tokens = _clamp_max_tokens("", limited_msgs, req.max_tokens)
        payload = {
            "model": OPENAI_MODEL,
            "messages": limited_msgs,
            "stream": False,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                j = r.json()

                # ★ 추가: 메타 태스크 응답도 think 제거
                try:
                    msg = j.get("choices", [{}])[0].get("message", {})
                    c = msg.get("content") or ""
                    msg["content"] = clean_llm_output(c)
                    j["choices"][0]["message"] = msg
                except Exception:
                    pass

                return _attach_trace(j, trace_id)
        except (httpx.RequestError, ValueError) as e:
            # 타임아웃/네트워크 장애는 200으로 안전하게 래핑해 돌려줌
            return _attach_trace({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "메타 태스크 처리 중 백엔드 응답 지연으로 실패했어요. 잠시 후 다시 시도해주세요."
                    },
                    "finish_reason": "stop"
                }],
                "error": {"type": e.__class__.__name__, "message": str(e)}
            }, trace_id)


    ctx_text = ""
    qa_json = None
    qa_items = []
    qa_urls: List[str] = []     # QA 경로 출처

    best_ctx = ""
    src_urls: List[str] = []
    ctx_items: list[dict] = []

    timeout = _httpx_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        spaces_hint = await _auto_pick_spaces(orig_user_msg, client)
        state = None
        reason = None
        if ROUTER_TRI_STATE:
            state, reason = _route_state(orig_user_msg, file_hint, spaces_hint, rag_followup, topic_shift)
            # LLM router override (skip only for hard RAG triggers)
            hard_required = bool(
                file_hint
                or spaces_hint
                or _CONF_HOST_RE.search(orig_user_msg or "")
                or _CONF_HINT_RE.search(orig_user_msg or "")
                or _PAGEID_HINT_RE.search(orig_user_msg or "")
            )
            if ROUTER_LLM_ROUTE and not hard_required:
                llm_state, llm_reason = await _llm_route_state(orig_user_msg, rag_followup, topic_shift)
                if llm_state:
                    state, reason = llm_state, llm_reason
                elif ROUTER_LLM_ROUTE_STRICT:
                    state, reason = "RAG_PREFERRED", "llm_fallback"
            if (
                ROUTER_PRECHECK_RAG
                and state == "NO_RAG"
                and not hard_required
                and not file_hint
                and not spaces_hint
            ):
                precheck_msgs = []
                if clean_user_msg:
                    precheck_msgs.append(clean_user_msg)
                if rewrite_q and rewrite_q != clean_user_msg:
                    precheck_msgs.append(rewrite_q)
                if variants:
                    precheck_msgs.extend(variants)
                precheck_msgs = list(dict.fromkeys(precheck_msgs))
                if await _precheck_rag_local(precheck_msgs):
                    state, reason = "RAG_PREFERRED", "precheck_rag"
            _dbg(f"route_state: trace_id={trace_id} state={state} reason={reason}")
        if ROUTER_TRI_STATE_ENFORCE and state == "NO_RAG" and not (file_hint or spaces_hint):
            content = await _llm_direct_answer(limited_msgs, req)
            if ROUTER_SHOW_CONTEXT_LABEL:
                content = "근거: 없음(LLM)\n" + content
            return _attach_trace({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
            }, trace_id)

        # [ADD] === Clarification 체크 (QA 전에 먼저 실행) ===
        if not clarification_choice_src:  # 이미 선택한 경우는 스킵
            try:
                clarify_payload = {"q": clean_user_msg, "k": 5, "sticky": False}
                if spaces_hint:
                    clarify_payload["spaces"] = spaces_hint
                _add_trace(clarify_payload, trace_id)
                clarify_resp = (await client.post(f"{RAG}/query", json=clarify_payload)).json()
                if clarify_resp.get("clarification_needed"):
                    candidates = clarify_resp.get("candidates", [])
                    message = clarify_resp.get("message", "문서를 선택해주세요.")
                    if candidates:
                        _save_clarification(trace_id, candidates)
                        content = _format_clarification_message(candidates, message)
                        _dbg(f"clarification_early: trace_id={trace_id} candidates={len(candidates)}")
                        return _attach_trace({
                            "id": f"cmpl-{uuid.uuid4()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop"
                            }],
                        }, trace_id)
            except Exception as e:
                _dbg(f"clarification_early_error: {e}")

        # === QA 경로 ===
        qa_json = None; qa_items = []; qa_urls = []
        if file_hint or force_mcp:
            _dbg("qa_skip_file_hint: true")
        else:
            for v in variants:
                #  QA 호출에도 spaces_hint 전달(기존 기능 유지, 정확도 ↑) 
                try:
                    p1 = {"q": v, "k": 5, "sticky": False}
                    if spaces_hint:
                        p1["spaces"] = spaces_hint
                    j1 = (await client.post(f"{RAG}/qa", json=p1)).json()
                except Exception:
                    j1 = {}
                try:
                    p2 = {"q": v, "k": 5, "sticky": True}
                    if spaces_hint:
                        p2["spaces"] = spaces_hint
                    j2 = (await client.post(f"{RAG}/qa", json=p2)).json()
                except Exception:
                    j2 = {}


                # 더 나은 쪽 선택
                best = max([j1, j2], key=_qa_score)
                if _qa_score(best) <= 0:
                    continue

                items = best.get("items") or []
                ctx_text = "\n\n".join(extract_texts(items))[:MAX_CTX_CHARS]
                ctx_text = mark_lonely_numbers_as_total(ctx_text)

                if not ctx_text.strip():
                    continue
                if (len(ctx_text) < ROUTER_QA_MIN_CTX_LEN) and not file_hint:
                    _dbg(f"qa_skip_short: ctx_len={len(ctx_text)} min={ROUTER_QA_MIN_CTX_LEN}")
                    continue
                if not (file_hint or is_good_context_for_qa(ctx_text) or is_relevant(orig_user_msg, ctx_text)):
                    continue

                qa_json  = best
                qa_items = best.get("items") or []
                qa_urls = _limit_urls(best.get("source_urls")) if best.get("source_urls") else _collect_urls_from_items(qa_items)
                _dbg(f"qa_pick: q='{v}' hits={best.get('hits')} items={len(qa_items)} ctx_len={len(ctx_text)}")
                break

    # 2-A) QA 성공
    if qa_json:
        primary_src = ""
        if qa_items:
            filtered_items, src = _prefer_single_source(qa_items, clean_user_msg)
            if src and _LOCAL_SRC_RE.search(src):
                primary_src = src
                if file_hint:
                    qa_items = filtered_items
                # local source should not inherit confluence URLs
                qa_urls = []
                if ROUTER_DEBUG:
                    _dbg(f"qa_source_filter: src='{primary_src}' items={len(qa_items)}")
        ctx_text = "\n\n".join(extract_texts(qa_items))[:MAX_CTX_CHARS]
        ctx_text = mark_lonely_numbers_as_total(ctx_text)
        if primary_src:
            top_kind = ""
            topn_items = qa_items[:ROUTER_TITLE_EXPAND_TOPN] if qa_items else []
            if qa_items:
                md = (qa_items[0].get("metadata") or {}) if isinstance(qa_items[0], dict) else {}
                top_kind = (md.get("kind") or qa_items[0].get("kind") or "")
            if topn_items:
                title_topn = sum(
                    1 for it in topn_items
                    if (((it.get("metadata") or {}) if isinstance(it, dict) else {}).get("kind")
                        or (it.get("kind") if isinstance(it, dict) else "")) in ROUTER_TITLE_EXPAND_KINDS
                )
                all_title_topn = (title_topn == len(topn_items))
            else:
                all_title_topn = False
            title_expand = ROUTER_TITLE_EXPAND and (all_title_topn or top_kind in ROUTER_TITLE_EXPAND_KINDS)
            focus_q = _focus_query_for_source(clean_user_msg)
            need_focus = bool(focus_q and focus_q != clean_user_msg)
            if need_focus or len(ctx_text) < ROUTER_QA_MIN_CTX_LEN or title_expand:
                try:
                    focus_q = focus_q or clean_user_msg
                    if ROUTER_DEBUG:
                        _dbg(f"qa_source_focus: q='{clean_user_msg}' focus='{focus_q}' src='{primary_src}'")
                    payload_src = {
                        "question": focus_q,
                        "k": ROUTER_TITLE_EXPAND_K,
                        "sticky": False,
                        "need_fallback": False,
                        "source": primary_src,
                    }
                    _add_trace(payload_src, trace_id)
                    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                        j_src = (await client.post(f"{RAG}/query", json=payload_src)).json()
                    items2 = (j_src.get("items") or j_src.get("contexts") or [])
                    if items2:
                        qa_items = items2
                        qa_urls = []
                        ctx_text = "\n\n".join(extract_texts(qa_items))[:MAX_CTX_CHARS]
                        ctx_text = mark_lonely_numbers_as_total(ctx_text)
                        if ROUTER_DEBUG:
                            _dbg(f"qa_source_requery: src='{primary_src}' ctx_len={len(ctx_text)} items={len(items2)}")
                except Exception:
                    pass
        if not file_hint:
            for it in qa_items or []:
                meta = it.get("metadata") or {}
                src = str(it.get("source") or meta.get("source") or "")
                if _LOCAL_SRC_RE.search(src):
                    file_hint = True
                    break
        ctx_items = qa_items
    # [CHANGE] 길이(80자) 허용 삭제 → 관련도/컨텍스트 품질만
    qa_ok = bool(ctx_text.strip()) and (
        file_hint or is_good_context_for_qa(ctx_text) or is_relevant(orig_user_msg, ctx_text)
    )
    if qa_ok and (len(ctx_text) < ROUTER_QA_MIN_CTX_LEN) and not file_hint:
        _dbg(f"qa_reject_short: ctx_len={len(ctx_text)} min={ROUTER_QA_MIN_CTX_LEN}")
        qa_ok = False
    if not qa_ok:
        if ROUTER_DEBUG:
            _dbg(f"qa_reject: ctx_len={len(ctx_text)} file_hint={file_hint} good_ctx={is_good_context_for_qa(ctx_text)} rel={is_relevant(orig_user_msg, ctx_text) if ctx_text else False}")
        qa_json = None

    if qa_json:
        if not file_hint and _items_have_local_source(qa_items):
            file_hint = True
        ctx_for_prompt = sanitize(ctx_text)
        mode = pick_answer_mode(orig_user_msg, ctx_for_prompt)

        sys_prefix = _build_system_prefix(mode)
        user_for_budget = next((m["content"] for m in reversed(limited_msgs) if m["role"]=="user"), "")
        ctx_for_prompt = _fit_ctx_to_budget(sys_prefix, user_for_budget, ctx_for_prompt)

        system_prompt = build_system_with_context(ctx_for_prompt, mode, force_summary=bool(file_hint and ctx_for_prompt))
        max_tokens = _clamp_max_tokens(system_prompt, limited_msgs, req.max_tokens)
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role":"system","content":system_prompt}] + limited_msgs,
            "stream": False,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                rj = r.json()
                raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                if ROUTER_DEBUG and not raw:
                    _dbg(f"qa_empty: status={r.status_code} keys={list(rj.keys())} err={rj.get('error')}")
                if not raw:
                    short_ctx = ctx_for_prompt[:2000]
                    payload["messages"] = [{"role":"system","content":build_final_only_prompt(short_ctx)}] + [{"role":"user","content": clean_user_msg}]
                    r2 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                    rj2 = r2.json()
                    raw = rj2.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            except (httpx.RequestError, ValueError) as e:
                print(f"[router] OPENAI chat error: {e}")
                raw = ""

        cleaned = clean_llm_output(raw)
        if ROUTER_DEBUG and not cleaned and raw:
            _dbg(f"qa_clean_empty: raw_prefix={repr(raw[:200])}")
        if cleaned.strip() == "인덱스에 근거 없음" and ctx_for_prompt:
            try:
                _dbg("qa_force_context_retry")
                payload["messages"] = [{"role":"system","content":build_force_context_prompt(ctx_for_prompt)}] + [{"role":"user","content": clean_user_msg}]
                async with httpx.AsyncClient(timeout=timeout) as retry_client:
                    r3 = await retry_client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                    rj3 = r3.json()
                raw = rj3.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                cleaned = clean_llm_output(raw)
            except Exception as e:
                _dbg(f"qa_force_context_error: {e}")
            if cleaned.strip() == "인덱스에 근거 없음":
                fallback = _fallback_summary_from_ctx(ctx_for_prompt)
                if fallback:
                    cleaned = fallback
        if _is_refusal_like(cleaned) and ctx_for_prompt:
            fallback = _fallback_summary_from_ctx(ctx_for_prompt)
            if fallback:
                cleaned = fallback
        content = sanitize(cleaned) or "인덱스에 근거 없음"
        _dbg(f"qa_answer: raw_len={len(raw)} content_len={len(content)}")

        full_ctx_for_check = sanitize(ctx_text)
        strict = bool(spaces_hint) and ROUTER_STRICT_RAG and not file_hint
        if strict and not supported_by_context(content, full_ctx_for_check):
            content = "인덱스에 근거 없음"
            qa_urls = []
        # 출처는 허용 호스트만(환경변수로 제어)
        if ROUTER_SHOW_SOURCES and content != "인덱스에 근거 없음":
            urls = _filter_urls_by_host(qa_urls)
            if urls:
                content += "\n\n출처:\n" + "\n".join(f"- {u}" for u in urls)

        if ROUTER_SHOW_CONTEXT_LABEL:
            content = f"근거: {_label_for_context(ctx_items or qa_items, qa_urls)}\n{content}"

        return _attach_trace({
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        }, trace_id)

    # ====== 2-B) QA 실패 → QUERY 경로 ======
    async with httpx.AsyncClient(timeout=timeout) as client:
        best_ctx_good=""; best_ctx_any=""
        best_urls_good=[]; best_urls_any=[]
        best_items_good=[]; best_items_any=[]
        used_q_good=None; used_q_any=None

        # [PATCH] /qa 호출 페이로드에 spaces 전달
        for v in variants:
            try:
                sticky1 = False
                sticky2 = True
                if page_id:
                    sticky1 = False
                    sticky2 = False
                payload1 = {"question": v, "k": 10, "sticky": sticky1, "need_fallback": bool(force_mcp)}
                if page_id:
                    payload1["page_id"] = page_id
                if file_hint_src and not force_mcp:
                    payload1["source"] = file_hint_src
                if not force_mcp:
                    _apply_src_filters(payload1, file_hint_src, meta_src_set)
                if spaces_hint:
                    payload1["spaces"] = spaces_hint
                # [ADD] Clarification choice 전달
                if clarification_choice_src:
                    payload1["clarification_choice"] = clarification_choice_src
                _add_trace(payload1, trace_id)
                j1 = await _query_with_wait(client, payload1, file_hint)
            except Exception:
                j1 = {}

            try:
                sticky1 = False
                sticky2 = True
                if page_id:
                    sticky1 = False
                    sticky2 = False
                payload2 = {"question": v, "k": 10, "sticky": sticky2, "need_fallback": bool(force_mcp)}
                if page_id:
                    payload2["page_id"] = page_id
                if file_hint_src and not force_mcp:
                    payload2["source"] = file_hint_src
                if not force_mcp:
                    _apply_src_filters(payload2, file_hint_src, meta_src_set)
                if spaces_hint:
                    payload2["spaces"] = spaces_hint
                # [ADD] Clarification choice 전달
                if clarification_choice_src:
                    payload2["clarification_choice"] = clarification_choice_src
                _add_trace(payload2, trace_id)
                j2 = await _query_with_wait(client, payload2, file_hint)
            except Exception:
                j2 = {}


            # 여기서 바로 평가/갱신 (바깥에 동일 코드 두지 말기)
            for qj in (j1, j2):
                # [ADD] Clarification 응답 처리
                if qj.get("clarification_needed"):
                    candidates = qj.get("candidates", [])
                    message = qj.get("message", "문서를 선택해주세요.")
                    if candidates:
                        _save_clarification(trace_id, candidates)
                        content = _format_clarification_message(candidates, message)
                        _dbg(f"clarification: trace_id={trace_id} candidates={len(candidates)}")
                        return _attach_trace({
                            "id": f"cmpl-{uuid.uuid4()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop"
                            }],
                        }, trace_id)

                direct_answer = (qj.get("direct_answer") or "").strip()
                if direct_answer:
                    if ROUTER_SHOW_CONTEXT_LABEL:
                        direct_answer = "근거: 없음(RAG)\n" + direct_answer
                    return _attach_trace({
                        "id": f"cmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": direct_answer},
                            "finish_reason": "stop"
                        }],
                    }, trace_id)
                items = (qj.get("items") or qj.get("contexts") or [])
                if file_hint and items:
                    filtered, primary_src = _prefer_single_source(items, clean_user_msg)
                    if primary_src and _LOCAL_SRC_RE.search(primary_src):
                        if ROUTER_DEBUG and len(filtered) < len(items):
                            _dbg(f"query_source_filter: src='{primary_src}' items={len(filtered)}")
                        items = filtered
                urls  = _limit_urls(qj.get("source_urls")) if qj.get("source_urls") else _collect_urls_from_items(items)
                if items and _items_have_local_source(items):
                    urls = []
                ctx_list = (qj.get("context_texts")
                            or [c.get("text","") for c in (qj.get("contexts") or [])]
                            or [it.get("text","") for it in (qj.get("items") or [])])
                if file_hint and items:
                    ctx_list = extract_texts(items)
                ctx = "\n\n---\n\n".join([t for t in ctx_list if t])[:MAX_CTX_CHARS]
                if ROUTER_DEBUG:
                    top = items[0] if items else {}
                    top_kind = (top.get("metadata") or {}).get("kind") or top.get("kind")
                    top_text = (top.get("text") or "") if isinstance(top, dict) else ""
                    _dbg(f"query_resp: q='{v}' hits={qj.get('hits')} items={len(items)} ctx_texts={len(ctx_list)} top_kind={top_kind} top_len={len(str(top_text))}")
                topn_items = items[:ROUTER_TITLE_EXPAND_TOPN] if items else []
                if topn_items:
                    title_topn = sum(
                        1 for it in topn_items
                        if (((it.get("metadata") or {}) if isinstance(it, dict) else {}).get("kind")
                            or (it.get("kind") if isinstance(it, dict) else "")) == "title"
                    )
                    all_title_topn = (title_topn == len(topn_items))
                else:
                    all_title_topn = False
                if items:
                    top = items[0]
                    md = (top.get("metadata") or {}) if isinstance(top, dict) else {}
                    top_kind = (md.get("kind") or top.get("kind") or "")
                    src = _get_item_source(top)
                else:
                    top_kind = ""
                    src = ""
                title_expand = ROUTER_TITLE_EXPAND and (all_title_topn or top_kind in ROUTER_TITLE_EXPAND_KINDS)
                if items and (len(ctx) < ROUTER_QA_MIN_CTX_LEN or title_expand):
                    if src and title_expand:
                        try:
                            focus_q = _focus_query_for_source(v) or v
                            if ROUTER_DEBUG:
                                _dbg(f"query_source_focus: q='{v}' focus='{focus_q}' src='{src}'")
                            payload_src = {"question": focus_q, "k": ROUTER_TITLE_EXPAND_K, "sticky": False, "need_fallback": False, "source": src}
                            _add_trace(payload_src, trace_id)
                            j_src = (await client.post(f"{RAG}/query", json=payload_src)).json()
                            items2 = (j_src.get("items") or j_src.get("contexts") or [])
                            urls2  = _limit_urls(j_src.get("source_urls")) if j_src.get("source_urls") else _collect_urls_from_items(items2)
                            ctx_list2 = (j_src.get("context_texts")
                                         or [c.get("text","") for c in (j_src.get("contexts") or [])]
                                         or [it.get("text","") for it in (j_src.get("items") or [])])
                            if file_hint and items2:
                                ctx_list2 = extract_texts(items2)
                            if items2 and _items_have_local_source(items2):
                                urls2 = []
                            ctx2 = "\n\n---\n\n".join([t for t in ctx_list2 if t])[:MAX_CTX_CHARS]
                            if len(ctx2) > len(ctx):
                                items = items2
                                urls = urls2
                                ctx_list = ctx_list2
                                ctx = ctx2
                                _dbg(f"query_source_boost: src='{src}' ctx_len={len(ctx2)} items={len(items2)}")
                        except Exception:
                            pass

                if len(ctx) > len(best_ctx_any):
                    best_ctx_any = ctx; best_urls_any = urls[:]; used_q_any = v; best_items_any = items[:]
                if is_good_context_for_qa(ctx) and len(ctx) > len(best_ctx_good):
                    best_ctx_good = ctx; best_urls_good = urls[:]; used_q_good = v; best_items_good = items[:]

        best_ctx = best_ctx_good or best_ctx_any
        src_urls = best_urls_good or best_urls_any
        ctx_items = best_items_good or best_items_any
        used_q_for_relevance = used_q_good if best_ctx_good else used_q_any
        core_terms = _core_query_terms(clean_user_msg, max_terms=4)
        if core_terms:
            core_items = _filter_items_by_keywords(ctx_items, core_terms)
            if core_items:
                core_ctx = "\n\n---\n\n".join(extract_texts(core_items))[:MAX_CTX_CHARS]
                if core_ctx:
                    ctx_items = core_items
                    best_ctx = core_ctx
                    if ROUTER_DEBUG:
                        _dbg(f"query_core_focus: ctx_len={len(best_ctx)} items={len(ctx_items)} terms={core_terms}")
        _dbg(f"query_pick: q='{used_q_for_relevance}' ctx_len={len(best_ctx)} urls={len(src_urls)}")

        # 게이트
        if best_ctx and not (file_hint or rag_followup or is_good_context_for_qa(best_ctx) or
                            is_relevant(used_q_for_relevance or orig_user_msg, best_ctx)):
            best_ctx = ""
            src_urls = []
            ctx_items = []

    if not best_ctx or len(best_ctx) < ROUTER_QA_MIN_CTX_LEN:
        if file_hint and ctx_items:
            if not best_ctx:
                ctx_list_short = extract_texts(ctx_items)
                best_ctx = "\n\n---\n\n".join([t for t in ctx_list_short if t])[:MAX_CTX_CHARS]
            _dbg(f"query_keep_short_ctx: ctx_len={len(best_ctx)} items={len(ctx_items)} reason=file_hint")
        else:
            if history_src and not force_mcp:
                try:
                    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                        resolved_src = await _resolve_source_from_index(client, history_src)
                        if resolved_src != history_src:
                            _dbg(f"query_history_source_resolve: src='{history_src}' -> '{resolved_src}'")
                        _dbg(f"query_history_source_try: src='{resolved_src}' q='{clean_user_msg}'")
                        payload = {"question": clean_user_msg, "k": 10, "sticky": False, "need_fallback": False, "source": resolved_src}
                        _add_trace(payload, trace_id)
                        j_hist = await client.post(f"{RAG}/query", json=payload)
                        j_hist = j_hist.json() if hasattr(j_hist, "json") else {}
                        items_hist = (j_hist.get("items") or j_hist.get("contexts") or [])
                        ctx_list_hist = (j_hist.get("context_texts")
                                         or [c.get("text","") for c in (j_hist.get("contexts") or [])]
                                         or [it.get("text","") for it in (j_hist.get("items") or [])])
                        if items_hist:
                            ctx_list_hist = extract_texts(items_hist)
                        ctx_hist = "\n\n---\n\n".join([t for t in ctx_list_hist if t])[:MAX_CTX_CHARS]
                        _dbg(f"query_history_source_resp: items={len(items_hist)} ctx_len={len(ctx_hist)}")
                        if not items_hist:
                            tokens = [t for t in _tokens(clean_user_msg) if t not in _STOPWORDS]
                            stem_terms = [t for t in _tokens(_file_stem_for_query(resolved_src)) if t not in _STOPWORDS]
                            overlap = [t for t in tokens if t in stem_terms]
                            if not overlap:
                                if ROUTER_DEBUG:
                                    _dbg(f"query_history_source_terms_skip: no overlap (tokens={tokens}, stem={stem_terms})")
                            else:
                                terms = list(dict.fromkeys(tokens + stem_terms))[:8]
                                if not terms and ROUTER_DEBUG:
                                    _dbg(f"query_history_source_terms: empty (msg={clean_user_msg!r})")
                                if terms:
                                    _dbg(f"query_history_source_terms: {terms}")
                                    items_hist, ctx_hist = await _query_source_with_terms(client, resolved_src, terms, k=20)
                                    _dbg(f"query_history_source_terms_resp: items={len(items_hist)} ctx_len={len(ctx_hist)}")
                            if not items_hist:
                                stem_q = _file_stem_for_query(resolved_src)
                                if stem_q and overlap:
                                    _dbg(f"query_history_source_fallback: q='{stem_q}'")
                                    items_hist, ctx_hist = await _query_source_with_terms(client, resolved_src, [stem_q], k=30)
                                    _dbg(f"query_history_source_fallback_resp: items={len(items_hist)} ctx_len={len(ctx_hist)}")
                        if items_hist and ctx_hist and (not best_ctx or len(ctx_hist) > len(best_ctx)):
                            best_ctx = ctx_hist
                            src_urls = []
                            ctx_items = items_hist
                            file_hint = True
                            _dbg(f"query_history_source: src='{resolved_src}' ctx_len={len(best_ctx)} items={len(items_hist)}")
                except Exception:
                    pass
        if file_hint and best_ctx:
            # already have context; no need to infer or fallback.
            pass
        if not force_mcp:
            inferred_src = _match_upload_source_by_query(clean_user_msg)
            if inferred_src:
                try:
                    _dbg(f"query_infer_source_try: src='{inferred_src}' q='{clean_user_msg}'")
                    payload = {"question": clean_user_msg, "k": 10, "sticky": False, "need_fallback": False, "source": inferred_src}
                    _add_trace(payload, trace_id)
                    async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                        j_inf = await client.post(f"{RAG}/query", json=payload)
                    j_inf = j_inf.json() if hasattr(j_inf, "json") else {}
                    items_inf = (j_inf.get("items") or j_inf.get("contexts") or [])
                    ctx_list_inf = (j_inf.get("context_texts")
                                    or [c.get("text","") for c in (j_inf.get("contexts") or [])]
                                    or [it.get("text","") for it in (j_inf.get("items") or [])])
                    if items_inf:
                        ctx_list_inf = extract_texts(items_inf)
                    ctx_inf = "\n\n---\n\n".join([t for t in ctx_list_inf if t])[:MAX_CTX_CHARS]
                    _dbg(f"query_infer_source_resp: items={len(items_inf)} ctx_len={len(ctx_inf)}")
                    if items_inf and ctx_inf and (not best_ctx or len(ctx_inf) > len(best_ctx)):
                        best_ctx = ctx_inf
                        src_urls = []
                        ctx_items = items_inf
                        _dbg(f"query_infer_source: src='{inferred_src}' ctx_len={len(best_ctx)} items={len(items_inf)}")
                except Exception:
                    pass

    if not best_ctx and file_hint and not history_src and not force_mcp:
        latest_src = _latest_upload_source()
        if latest_src:
            try:
                fallback_q = _file_stem_for_query(latest_src) or clean_user_msg
                _dbg(f"query_latest_source_try: src='{latest_src}' q='{fallback_q}'")
                async with httpx.AsyncClient(timeout=_httpx_timeout()) as client:
                    resolved_src = await _wait_for_index_source(client, latest_src, ROUTER_INGEST_WAIT_SEC)
                    src = resolved_src or latest_src
                    payload = {"question": fallback_q, "k": 10, "sticky": False, "need_fallback": False, "source": src}
                    _add_trace(payload, trace_id)
                    j_latest = await _query_with_wait(client, payload, file_hint)
                items_latest = (j_latest.get("items") or j_latest.get("contexts") or [])
                ctx_list_latest = (j_latest.get("context_texts")
                                   or [c.get("text","") for c in (j_latest.get("contexts") or [])]
                                   or [it.get("text","") for it in (j_latest.get("items") or [])])
                if items_latest:
                    ctx_list_latest = extract_texts(items_latest)
                ctx_latest = "\n\n---\n\n".join([t for t in ctx_list_latest if t])[:MAX_CTX_CHARS]
                _dbg(f"query_latest_source_resp: items={len(items_latest)} ctx_len={len(ctx_latest)}")
                if items_latest and (ctx_latest or _ctx_ready_for_file(items_latest, ctx_latest)):
                    best_ctx = ctx_latest
                    src_urls = []
                    ctx_items = items_latest
                    _dbg(f"query_latest_source: src='{latest_src}' ctx_len={len(best_ctx)} items={len(items_latest)}")
            except Exception:
                pass

    if not best_ctx:
        if force_mcp:
            allow_llm = False
        elif ROUTER_TRI_STATE_ENFORCE and state == "RAG_REQUIRED":
            allow_llm = False
        else:
            allow_llm = _allow_llm_fallback(orig_user_msg, file_hint, spaces_hint, rag_followup, topic_shift)
        if not allow_llm:
            return _attach_trace({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "근거: 없음(RAG)\n문서 컨텍스트가 없어 답변할 수 없습니다."},
                    "finish_reason": "stop"
                }],
            }, trace_id)
        if spaces_hint:
            return _attach_trace({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "인덱스에 근거 없음"},
                    "finish_reason": "stop"
                }],
            }, trace_id)

        content = await _llm_direct_answer(limited_msgs, req)
        content = await _ensure_korean(content, req)

        if ROUTER_SHOW_CONTEXT_LABEL:
            content = "근거: 없음(LLM)\n" + content

        return _attach_trace({
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
        }, trace_id)

    
    # ===== QUERY 경로 LLM 호출 정리본 =====
    ctx_text = best_ctx
    ctx_text = mark_lonely_numbers_as_total(ctx_text)
    full_ctx_for_check = sanitize(ctx_text)

    # 모드 결정은 트림 전 컨텍스트로
    mode = pick_answer_mode(orig_user_msg, full_ctx_for_check)

    # 예산 계산에 필요한 메시지/프리픽스 준비
    sys_prefix = _build_system_prefix(mode)
    user_for_budget = next((m["content"] for m in reversed(limited_msgs) if m["role"] == "user"), "")

    # 컨텍스트를 토큰 예산에 맞춰 컷
    ctx_for_prompt = _fit_ctx_to_budget(sys_prefix, user_for_budget, full_ctx_for_check)
    system_prompt = build_system_with_context(ctx_for_prompt, mode, force_summary=bool(file_hint and ctx_for_prompt))
    max_tokens = _clamp_max_tokens(system_prompt, limited_msgs, req.max_tokens)

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + limited_msgs,
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
    }


    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            rj = r.json()
            raw = rj.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            if ROUTER_DEBUG and not raw:
                _dbg(f"query_empty: status={r.status_code} keys={list(rj.keys())} err={rj.get('error')}")
            if raw.strip() == "인덱스에 근거 없음":
                _dbg("query_retry_due_to_basis_none")
                short_ctx = ctx_for_prompt[:2000]
                payload["messages"] = [{"role":"system","content":build_final_only_prompt(short_ctx)}] + [{"role":"user","content": clean_user_msg}]
                r2 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                rj2 = r2.json()
                raw = rj2.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            if not raw:
                short_ctx = ctx_for_prompt[:2000]
                payload["messages"] = [{"role":"system","content":build_final_only_prompt(short_ctx)}] + [{"role":"user","content": clean_user_msg}]
                r2 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
                rj2 = r2.json()
                raw = rj2.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except (httpx.RequestError, ValueError) as e:
            print(f"[router] OPENAI chat error: {e}")
            raw = ""

    cleaned = clean_llm_output(raw)
    if ROUTER_DEBUG and not cleaned and raw:
        _dbg(f"query_clean_empty: raw_prefix={repr(raw[:200])}")
    if cleaned.strip() == "인덱스에 근거 없음" and full_ctx_for_check:
        try:
            _dbg("query_force_context_retry")
            payload["messages"] = [{"role":"system","content":build_force_context_prompt(ctx_for_prompt)}] + [{"role":"user","content": clean_user_msg}]
            r3 = await client.post(f"{OPENAI}/chat/completions", json=payload, headers=_openai_headers())
            rj3 = r3.json()
            raw = rj3.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            cleaned = clean_llm_output(raw)
        except Exception:
            pass
        if cleaned.strip() == "인덱스에 근거 없음":
            fallback = _fallback_summary_from_ctx(ctx_for_prompt)
            if fallback:
                cleaned = fallback
    if _is_refusal_like(cleaned) and full_ctx_for_check:
        fallback = _fallback_summary_from_ctx(ctx_for_prompt)
        if fallback:
            cleaned = fallback
    content = sanitize(cleaned) or "인덱스에 근거 없음"
    content = await _ensure_korean(content, req)
    if not file_hint and best_ctx and _LOCAL_SRC_RE.search(best_ctx):
        file_hint = True

    #  STRICT는 스페이스 힌트 있을 때만, 출처 필터 적용
    strict = bool(spaces_hint) and ROUTER_STRICT_RAG and not file_hint
    if strict and not supported_by_context(content, full_ctx_for_check):
        content = "인덱스에 근거 없음"
        src_urls = []

    if ROUTER_SHOW_SOURCES and content != "인덱스에 근거 없음":
        urls = _filter_urls_by_host(src_urls)
        if urls:
            content += "\n\n출처:\n" + "\n".join(f"- {u}" for u in urls)

    if ROUTER_SHOW_CONTEXT_LABEL:
        content = f"근거: {_label_for_context(ctx_items or qa_items, src_urls)}\n{content}"

    return _attach_trace({
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
    }, trace_id)
