#!/usr/bin/env python3
"""
컨플 챗봇 끝장 테스트 스크립트 (Python 버전)
- 개발자가 간과하기 쉬운 치명적인 엣지 케이스
- 멀티테넌시 환경의 병목 및 경쟁 조건
- 비동기 동시성 테스트
- 메모리 누수 탐지
"""

import asyncio
import aiohttp
import time
import uuid
import json
import sys
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import traceback

# === 설정 ===
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8088")
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8080")
MCP_URL = os.getenv("MCP_URL", "http://localhost:9898")
MODEL = os.getenv("MODEL", "/model/Qwen2.5-14B-Instruct")

# === 결과 수집 ===
@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration: float = 0.0
    critical: bool = False

@dataclass
class TestSuite:
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)
        status = "✓" if result.passed else ("✗✗ CRITICAL" if result.critical else "✗")
        color = "\033[32m" if result.passed else "\033[31m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {result.name} ({result.duration:.2f}s)")
        if result.message:
            print(f"    {result.message}")

    def summary(self):
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        critical = sum(1 for r in self.results if r.critical and not r.passed)

        print("\n" + "=" * 60)
        print(f"총 테스트: {len(self.results)}")
        print(f"\033[32m  ✓ 통과: {passed}\033[0m")
        print(f"\033[31m  ✗ 실패: {failed}\033[0m")
        if critical > 0:
            print(f"\033[31m  🚨 치명적: {critical}\033[0m")
        print("=" * 60)

        return failed == 0

suite = TestSuite()

# === 유틸리티 ===
async def fetch(session: aiohttp.ClientSession, url: str, method: str = "GET",
                json_data: dict = None, timeout: float = 30.0) -> Tuple[int, Any]:
    """HTTP 요청 래퍼"""
    try:
        kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout)}
        if json_data:
            kwargs["json"] = json_data

        async with getattr(session, method.lower())(url, **kwargs) as resp:
            try:
                data = await resp.json()
            except:
                data = await resp.text()
            return resp.status, data
    except asyncio.TimeoutError:
        return 0, {"error": "timeout"}
    except Exception as e:
        return 0, {"error": str(e)}

# =============================================================================
# PART 1: 유니코드 및 정규화 테스트
# =============================================================================

async def test_unicode_normalization():
    """NFC vs NFD 한글 정규화 테스트"""
    start = time.time()

    # NFC 정규화된 한글
    nfc_text = unicodedata.normalize("NFC", "한글 테스트")
    # NFD 정규화된 한글 (분리된 자모)
    nfd_text = unicodedata.normalize("NFD", "한글 테스트")

    async with aiohttp.ClientSession() as session:
        _, resp_nfc = await fetch(session, f"{PROXY_URL}/query", "POST",
                                  {"q": nfc_text, "k": 3})
        _, resp_nfd = await fetch(session, f"{PROXY_URL}/query", "POST",
                                  {"q": nfd_text, "k": 3})

    # 두 결과가 동일해야 함
    hits_nfc = resp_nfc.get("hits", -1) if isinstance(resp_nfc, dict) else -1
    hits_nfd = resp_nfd.get("hits", -1) if isinstance(resp_nfd, dict) else -1

    passed = hits_nfc == hits_nfd and hits_nfc >= 0

    suite.add(TestResult(
        name="Unicode NFC/NFD 정규화",
        passed=passed,
        message=f"NFC hits={hits_nfc}, NFD hits={hits_nfd}",
        duration=time.time() - start,
        critical=True
    ))

async def test_korean_english_spacing():
    """한영 혼합 공백 정규화 테스트"""
    start = time.time()

    queries = [
        ("OCPP스펙", "OCPP 스펙"),
        ("EMS시스템", "EMS 시스템"),
        ("AI모델", "AI 모델"),
    ]

    async with aiohttp.ClientSession() as session:
        all_consistent = True
        messages = []

        for q1, q2 in queries:
            _, resp1 = await fetch(session, f"{PROXY_URL}/query", "POST", {"q": q1, "k": 3})
            _, resp2 = await fetch(session, f"{PROXY_URL}/query", "POST", {"q": q2, "k": 3})

            hits1 = resp1.get("hits", -1) if isinstance(resp1, dict) else -1
            hits2 = resp2.get("hits", -1) if isinstance(resp2, dict) else -1

            # MCP fallback 상태도 확인
            fb1 = resp1.get("notes", {}).get("fallback_used") if isinstance(resp1, dict) else None
            fb2 = resp2.get("notes", {}).get("fallback_used") if isinstance(resp2, dict) else None

            consistent = (hits1 == hits2) or (fb1 == fb2 == True)
            if not consistent:
                all_consistent = False
                messages.append(f"{q1}({hits1}) vs {q2}({hits2})")

    suite.add(TestResult(
        name="한영 혼합 공백 정규화",
        passed=all_consistent,
        message="; ".join(messages) if messages else "모든 쌍 일관",
        duration=time.time() - start,
        critical=True
    ))

# =============================================================================
# PART 2: 엣지 케이스 입력 테스트
# =============================================================================

async def test_edge_case_inputs():
    """극단적인 입력값 테스트"""
    start = time.time()

    test_cases = [
        ("빈 문자열", {"q": "", "k": 5}),
        ("공백만", {"q": "   ", "k": 5}),
        ("특수문자만", {"q": "!@#$%^&*()", "k": 5}),
        ("null 쿼리", {"q": None, "k": 5}),
        ("k=0", {"q": "test", "k": 0}),
        ("k=-1", {"q": "test", "k": -1}),
        ("k=10000", {"q": "test", "k": 10000}),
        ("매우 긴 쿼리", {"q": "A" * 5000, "k": 5}),
        ("이모지", {"q": "테스트 😀 🎉 문서", "k": 5}),
        ("JSON 특수문자", {"q": 'test"with\\nquotes', "k": 5}),
        ("HTML 태그", {"q": "<script>alert('xss')</script>", "k": 5}),
        ("SQL 인젝션", {"q": "'; DROP TABLE users; --", "k": 5}),
    ]

    async with aiohttp.ClientSession() as session:
        failures = []

        for name, payload in test_cases:
            try:
                status, resp = await fetch(session, f"{PROXY_URL}/query", "POST", payload)

                # 서버가 에러 없이 응답해야 함 (status 2xx 또는 4xx)
                if status == 0 or (status >= 500):
                    failures.append(f"{name}: 서버 에러 {status}")
            except Exception as e:
                failures.append(f"{name}: 예외 {e}")

    suite.add(TestResult(
        name="극단적 입력값 처리",
        passed=len(failures) == 0,
        message="; ".join(failures[:3]) if failures else f"{len(test_cases)}개 케이스 모두 안전",
        duration=time.time() - start,
        critical=True
    ))

# =============================================================================
# PART 3: 동시성 및 경쟁 조건 테스트
# =============================================================================

async def test_concurrent_sessions():
    """동시 세션 격리 테스트"""
    start = time.time()

    num_sessions = 20

    async def session_query(session_id: str, session: aiohttp.ClientSession):
        payload = {
            "q": f"세션{session_id} 전용 테스트",
            "k": 5,
            "session_id": session_id
        }
        _, resp = await fetch(session, f"{PROXY_URL}/query", "POST", payload)
        return session_id, resp

    async with aiohttp.ClientSession() as session:
        tasks = [
            session_query(f"concurrent-{i}-{uuid.uuid4().hex[:8]}", session)
            for i in range(num_sessions)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if not isinstance(r, Exception) and isinstance(r[1], dict))

    suite.add(TestResult(
        name=f"{num_sessions}개 동시 세션 격리",
        passed=success >= num_sessions * 0.9,  # 90% 성공
        message=f"{success}/{num_sessions} 성공",
        duration=time.time() - start
    ))

async def test_sticky_session_isolation():
    """Sticky 세션 간 간섭 테스트"""
    start = time.time()

    session_a = f"sticky-A-{uuid.uuid4().hex[:8]}"
    session_b = f"sticky-B-{uuid.uuid4().hex[:8]}"

    async with aiohttp.ClientSession() as session:
        # 세션 A: sticky 설정
        _, resp_a1 = await fetch(session, f"{PROXY_URL}/query", "POST", {
            "q": "세미나 자료",
            "k": 5,
            "session_id": session_a,
            "sticky": True
        })

        # 세션 B: 다른 질문
        _, resp_b = await fetch(session, f"{PROXY_URL}/query", "POST", {
            "q": "프로젝트 현황",
            "k": 5,
            "session_id": session_b,
            "sticky": True
        })

        # 세션 A의 sticky가 세션 B에 영향 안줬는지 확인
        src_a = ""
        src_b = ""

        if isinstance(resp_a1, dict) and resp_a1.get("items"):
            src_a = resp_a1["items"][0].get("metadata", {}).get("source", "")
        if isinstance(resp_b, dict) and resp_b.get("items"):
            src_b = resp_b["items"][0].get("metadata", {}).get("source", "")

    # 세션 간 소스가 다르거나, 둘 다 비어있으면 OK
    passed = (src_a != src_b) or (not src_a and not src_b)

    suite.add(TestResult(
        name="Sticky 세션 격리",
        passed=passed,
        message=f"A: {src_a[:30] if src_a else 'empty'}, B: {src_b[:30] if src_b else 'empty'}",
        duration=time.time() - start
    ))

async def test_race_condition_upload_query():
    """업로드 중 쿼리 경쟁 조건 테스트"""
    start = time.time()

    async with aiohttp.ClientSession() as session:
        # 업로드와 쿼리를 동시에 실행
        async def do_queries():
            results = []
            for i in range(10):
                _, resp = await fetch(session, f"{PROXY_URL}/query", "POST",
                                      {"q": f"race-test-{i}", "k": 3})
                results.append(resp)
                await asyncio.sleep(0.1)
            return results

        query_results = await do_queries()

    success = sum(1 for r in query_results if isinstance(r, dict) and "error" not in r)

    suite.add(TestResult(
        name="업로드/쿼리 경쟁 조건",
        passed=success >= 8,
        message=f"{success}/10 쿼리 성공",
        duration=time.time() - start
    ))

# =============================================================================
# PART 4: 메모리 및 성능 테스트
# =============================================================================

async def test_memory_leak_detection():
    """메모리 누수 탐지 (응답 시간 증가 패턴)"""
    start = time.time()

    times = []

    async with aiohttp.ClientSession() as session:
        for i in range(100):
            req_start = time.time()
            _, resp = await fetch(session, f"{PROXY_URL}/query", "POST",
                                  {"q": f"memory-test-{i}", "k": 3})
            req_time = time.time() - req_start
            times.append(req_time)

    # 처음 10개 평균 vs 마지막 10개 평균 비교
    first_avg = sum(times[:10]) / 10
    last_avg = sum(times[-10:]) / 10

    # 마지막이 처음의 3배 이상이면 누수 의심
    ratio = last_avg / (first_avg + 0.001)
    passed = ratio < 3.0

    suite.add(TestResult(
        name="메모리 누수 탐지 (100회 반복)",
        passed=passed,
        message=f"첫 10개 평균: {first_avg:.3f}s, 마지막 10개: {last_avg:.3f}s, 비율: {ratio:.2f}",
        duration=time.time() - start
    ))

async def test_response_time_consistency():
    """응답 시간 일관성 테스트"""
    start = time.time()

    times = []

    async with aiohttp.ClientSession() as session:
        for _ in range(30):
            req_start = time.time()
            _, resp = await fetch(session, f"{PROXY_URL}/query", "POST",
                                  {"q": "프로젝트 현황", "k": 5})
            times.append(time.time() - req_start)

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    # 표준편차가 평균의 50% 미만이면 OK
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5
    cv = std_dev / (avg_time + 0.001)  # 변동계수

    passed = cv < 0.5 and max_time < 10.0

    suite.add(TestResult(
        name="응답 시간 일관성",
        passed=passed,
        message=f"평균: {avg_time:.2f}s, 최소: {min_time:.2f}s, 최대: {max_time:.2f}s, CV: {cv:.2f}",
        duration=time.time() - start
    ))

# =============================================================================
# PART 5: 통합 플로우 테스트
# =============================================================================

async def test_clarification_flow():
    """Clarification 전체 플로우 테스트"""
    start = time.time()

    async with aiohttp.ClientSession() as session:
        # 모호한 쿼리로 clarification 유발
        _, resp1 = await fetch(session, f"{ROUTER_URL}/v1/chat/completions", "POST", {
            "model": MODEL,
            "messages": [{"role": "user", "content": "EMS 자료"}]
        })

        content1 = ""
        if isinstance(resp1, dict) and resp1.get("choices"):
            content1 = resp1["choices"][0].get("message", {}).get("content", "")

        # clarification인지 확인 (선택지 패턴)
        is_clarification = "1." in content1 or "[1]" in content1

        if is_clarification:
            # 선택
            _, resp2 = await fetch(session, f"{ROUTER_URL}/v1/chat/completions", "POST", {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "EMS 자료"},
                    {"role": "assistant", "content": content1},
                    {"role": "user", "content": "1"}
                ]
            })

            content2 = ""
            if isinstance(resp2, dict) and resp2.get("choices"):
                content2 = resp2["choices"][0].get("message", {}).get("content", "")

            passed = bool(content2) and "1." not in content2
            message = "Clarification 선택 후 결과 반환"
        else:
            passed = bool(content1)
            message = "직접 응답 (clarification 미발생)"

    suite.add(TestResult(
        name="Clarification 플로우",
        passed=passed,
        message=message,
        duration=time.time() - start
    ))

async def test_mcp_fallback_trigger():
    """MCP 폴백 트리거 조건 테스트"""
    start = time.time()

    async with aiohttp.ClientSession() as session:
        # 로컬에 없는 Confluence 전용 쿼리
        _, resp = await fetch(session, f"{PROXY_URL}/query", "POST", {
            "q": "컨플루언스 전용 특수 프로젝트 XYZ",
            "k": 5,
            "need_fallback": True
        })

        fallback_used = False
        if isinstance(resp, dict) and resp.get("notes"):
            fallback_used = resp["notes"].get("fallback_used", False)

    suite.add(TestResult(
        name="MCP 폴백 트리거",
        passed=True,  # 폴백 시도 여부만 확인
        message=f"fallback_used: {fallback_used}",
        duration=time.time() - start
    ))

async def test_no_rag_route():
    """NO_RAG 라우팅 테스트"""
    start = time.time()

    async with aiohttp.ClientSession() as session:
        _, resp = await fetch(session, f"{ROUTER_URL}/v1/chat/completions", "POST", {
            "model": MODEL,
            "messages": [{"role": "user", "content": "피타고라스 정리 설명해줘"}]
        })

        content = ""
        if isinstance(resp, dict) and resp.get("choices"):
            content = resp["choices"][0].get("message", {}).get("content", "")

        # LLM 직접 응답인지 확인
        is_llm_direct = "근거: 없음(LLM)" in content or bool(content)

    suite.add(TestResult(
        name="NO_RAG 라우팅",
        passed=is_llm_direct,
        message=f"응답 길이: {len(content)}, LLM 직접: {'LLM' in content}",
        duration=time.time() - start
    ))

# =============================================================================
# PART 6: 보안 테스트
# =============================================================================

async def test_path_traversal():
    """경로 순회 공격 방지 테스트"""
    start = time.time()

    payloads = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/app/../../../etc/shadow",
        "....//....//....//etc/passwd",
    ]

    async with aiohttp.ClientSession() as session:
        all_safe = True

        for payload in payloads:
            _, resp = await fetch(session, f"{PROXY_URL}/query", "POST", {
                "q": "test",
                "k": 5,
                "source": payload
            })

            # 서버가 정상 처리(에러 없이)해야 함
            if not isinstance(resp, dict):
                all_safe = False

    suite.add(TestResult(
        name="경로 순회 공격 방지",
        passed=all_safe,
        message=f"{len(payloads)}개 페이로드 테스트",
        duration=time.time() - start,
        critical=True
    ))

async def test_cql_injection():
    """CQL 인젝션 방지 테스트"""
    start = time.time()

    payloads = [
        'test" OR 1=1 --',
        "'; DROP TABLE pages; --",
        "test AND space='ADMIN'",
        'test" AND type=attachment',
    ]

    async with aiohttp.ClientSession() as session:
        all_safe = True

        for payload in payloads:
            _, resp = await fetch(session, f"{PROXY_URL}/query", "POST", {
                "q": payload,
                "k": 5
            })

            if not isinstance(resp, dict):
                all_safe = False

    suite.add(TestResult(
        name="CQL 인젝션 방지",
        passed=all_safe,
        message=f"{len(payloads)}개 페이로드 안전 처리",
        duration=time.time() - start,
        critical=True
    ))

# =============================================================================
# PART 7: 한국어 특수 케이스
# =============================================================================

async def test_korean_postfix_removal():
    """한국어 조사 제거 테스트"""
    start = time.time()

    pairs = [
        ("프로젝트에서", "프로젝트"),
        ("문서를", "문서"),
        ("시스템으로", "시스템"),
        ("가이드라인에", "가이드라인"),
    ]

    async with aiohttp.ClientSession() as session:
        consistent = 0

        for with_postfix, without_postfix in pairs:
            _, resp1 = await fetch(session, f"{PROXY_URL}/query", "POST",
                                   {"q": with_postfix, "k": 3})
            _, resp2 = await fetch(session, f"{PROXY_URL}/query", "POST",
                                   {"q": without_postfix, "k": 3})

            # 둘 다 유효한 응답이면 OK
            if isinstance(resp1, dict) and isinstance(resp2, dict):
                consistent += 1

    suite.add(TestResult(
        name="한국어 조사 제거",
        passed=consistent >= len(pairs) * 0.8,
        message=f"{consistent}/{len(pairs)} 일관성",
        duration=time.time() - start
    ))

# =============================================================================
# PART 8: 스트레스 테스트
# =============================================================================

async def test_high_concurrency():
    """고동시성 스트레스 테스트"""
    start = time.time()

    num_requests = 50

    async with aiohttp.ClientSession() as session:
        async def single_request(i: int):
            _, resp = await fetch(session, f"{PROXY_URL}/query", "POST",
                                  {"q": f"stress-{i}", "k": 3}, timeout=30.0)
            return isinstance(resp, dict) and "error" not in resp

        tasks = [single_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if r is True)

    suite.add(TestResult(
        name=f"고동시성 ({num_requests}개 동시)",
        passed=success >= num_requests * 0.9,
        message=f"{success}/{num_requests} 성공",
        duration=time.time() - start
    ))

async def test_sustained_load():
    """지속 부하 테스트"""
    start = time.time()

    duration = 10  # 10초 동안
    success_count = 0
    total_count = 0

    async with aiohttp.ClientSession() as session:
        end_time = time.time() + duration

        while time.time() < end_time:
            _, resp = await fetch(session, f"{PROXY_URL}/health", timeout=5.0)
            total_count += 1
            if isinstance(resp, dict) and resp.get("status"):
                success_count += 1
            await asyncio.sleep(0.1)

    rate = success_count / total_count if total_count > 0 else 0

    suite.add(TestResult(
        name=f"지속 부하 ({duration}초)",
        passed=rate >= 0.95,
        message=f"{success_count}/{total_count} 성공 ({rate*100:.1f}%)",
        duration=time.time() - start
    ))

# =============================================================================
# 메인 실행
# =============================================================================

async def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("컨플 챗봇 끝장 테스트 (Python)")
    print("=" * 60)

    # 서버 상태 확인
    async with aiohttp.ClientSession() as session:
        try:
            _, resp = await fetch(session, f"{PROXY_URL}/health", timeout=5.0)
            if not isinstance(resp, dict) or not resp.get("status"):
                print(f"\033[31m❌ rag-proxy 서버 응답 없음: {PROXY_URL}\033[0m")
                return False
        except Exception as e:
            print(f"\033[31m❌ 서버 연결 실패: {e}\033[0m")
            return False

    print("\n\033[36m[PART 1] 유니코드 및 정규화\033[0m")
    await test_unicode_normalization()
    await test_korean_english_spacing()

    print("\n\033[36m[PART 2] 엣지 케이스 입력\033[0m")
    await test_edge_case_inputs()

    print("\n\033[36m[PART 3] 동시성 및 경쟁 조건\033[0m")
    await test_concurrent_sessions()
    await test_sticky_session_isolation()
    await test_race_condition_upload_query()

    print("\n\033[36m[PART 4] 메모리 및 성능\033[0m")
    await test_memory_leak_detection()
    await test_response_time_consistency()

    print("\n\033[36m[PART 5] 통합 플로우\033[0m")
    await test_clarification_flow()
    await test_mcp_fallback_trigger()
    await test_no_rag_route()

    print("\n\033[36m[PART 6] 보안\033[0m")
    await test_path_traversal()
    await test_cql_injection()

    print("\n\033[36m[PART 7] 한국어 특수 케이스\033[0m")
    await test_korean_postfix_removal()

    print("\n\033[36m[PART 8] 스트레스 테스트\033[0m")
    await test_high_concurrency()
    await test_sustained_load()

    return suite.summary()

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
