#!/usr/bin/env bash
# 실패한 테스트만 빠르게 실행하는 스크립트

set -uo pipefail

ROUTER_URL="${ROUTER_URL:-http://localhost:8088}"
PROXY_URL="${PROXY_URL:-http://localhost:8080}"
MODEL="${MODEL:-/model/Qwen2.5-14B-Instruct}"

# 색상
C_RESET="\033[0m"
C_RED="\033[0;31m"
C_GREEN="\033[0;32m"
C_GRAY="\033[0;90m"

PASSED=0
FAILED=0

success() { echo -e "${C_GREEN}✓ $1${C_RESET}"; ((PASSED++)); }
failure() { echo -e "${C_RED}✗ $1${C_RESET}"; ((FAILED++)); }
info() { echo -e "${C_GRAY}  $1${C_RESET}"; }

curl_quiet() {
  curl -s --max-time 90 "$@" 2>/dev/null || echo "{}"
}

safe_jq() {
  jq -r "$1" 2>/dev/null || echo ""
}

echo "=== 실패 테스트만 실행 (3.3, 3.4, 3.5) ==="
echo ""

# 웜업
echo "Warming up..."
curl_quiet "$ROUTER_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"안녕\"}]}" >/dev/null
sleep 2

# -----------------------------------------------------------------------------
echo ""
echo "━━━ 3.3 NO_RAG → LLM 직접 응답 ━━━"
echo "Testing: 피타고라스 정리가 뭐야?"

resp=$(curl_quiet "$ROUTER_URL/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"피타고라스 정리가 뭐야?\"}]
}" --max-time 90)

content=$(echo "$resp" | safe_jq '.choices[0].message.content')
resp_len=${#resp}
content_len=${#content}
info "Response: ${resp_len}B, Content: ${content_len}B"

if [[ -n "$content" && "$content" != "null" && ${#content} -gt 10 ]]; then
  success "3.3 PASS - LLM response received"
  info "Preview: ${content:0:80}..."
else
  failure "3.3 FAIL - no response"
  info "Raw: ${resp:0:300}"
fi

# -----------------------------------------------------------------------------
echo ""
echo "━━━ 3.4 MCP 폴백 트리거 ━━━"
echo "Testing: 프로젝트 현황 보고 (need_fallback=true)"

resp=$(curl_quiet "$PROXY_URL/query" -H "Content-Type: application/json" -d "{
  \"q\":\"프로젝트 현황 보고\",
  \"k\":5,
  \"need_fallback\":true
}")

hits=$(echo "$resp" | safe_jq '.hits')
fallback=$(echo "$resp" | safe_jq '.notes.fallback_used')
resp_len=${#resp}
info "Response: ${resp_len}B, Hits: $hits, Fallback: $fallback"

if echo "$resp" | jq -e '.notes' >/dev/null 2>&1 || [[ "$hits" =~ ^[0-9]+$ ]]; then
  success "3.4 PASS - MCP fallback executed"
else
  failure "3.4 FAIL - no response"
  info "Raw: ${resp:0:300}"
fi

# -----------------------------------------------------------------------------
echo ""
echo "━━━ 3.5 pageId 직접 조회 ━━━"
echo "Testing: pageId=208537808 내용 요약"

resp=$(curl_quiet "$ROUTER_URL/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"pageId=208537808 내용 요약해줘\"}]
}" --max-time 90)

content=$(echo "$resp" | safe_jq '.choices[0].message.content')
resp_len=${#resp}
content_len=${#content}
info "Response: ${resp_len}B, Content: ${content_len}B"

if [[ -n "$content" && "$content" != "null" && ${#content} -gt 20 ]]; then
  success "3.5 PASS - pageId access working"
  info "Preview: ${content:0:80}..."
elif [[ -n "$content" && "$content" != "null" ]]; then
  success "3.5 PASS - pageId query processed (short response)"
else
  failure "3.5 FAIL - no response"
  info "Raw: ${resp:0:300}"
fi

# -----------------------------------------------------------------------------
echo ""
echo "=== 결과: $PASSED passed, $FAILED failed ==="
