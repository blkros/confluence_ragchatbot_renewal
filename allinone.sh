#!/usr/bin/env bash
set -euo pipefail

# === 수정 필요 ===
FILE_SRC="/app/uploads/3c67e1f9-f055-4be4-879e-d340be414b15_국세청_2025년 개정세법 요약.pdf"
PAGE_ID="268538899"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need curl
need jq

json() { jq -C . 2>/dev/null || cat; }

curl_timed() {
  local url="$1"
  local data="$2"
  local tmp timing
  tmp=$(mktemp)
  timing=$(curl -s -o "$tmp" -w "http=%{http_code} time_total=%{time_total}s" \
    -H "Content-Type: application/json" -d "$data" "$url")
  cat "$tmp"
  echo "# $timing" >&2
  rm -f "$tmp"
}

section() { echo -e "\n== $1 =="; }

section "Health"
curl -s http://localhost:8088/v1/models | json
curl -s http://localhost:8080/health | json
curl -s http://localhost:9898/health | json

section "Index"
if curl -s -f http://localhost:8080/index/stats >/dev/null; then
  curl -s http://localhost:8080/index/stats | json
  curl -s http://localhost:8080/index/sources | jq '.items[:5]'
else
  echo "index endpoints not available" >&2
fi

section "Rag-proxy query (no filter)"
resp=$(curl_timed "http://localhost:8080/query" \
  "{\"q\":\"OCPP 요약\",\"k\":5}")
echo "$resp" | jq -r '
  "hits=\(.hits)",
  "scores=" + ([(.items[]?.score // .contexts[]?.score // .items[]?.distance // .contexts[]?.distance)] | tostring),
  "notes=" + ((.notes // {}) | tostring)
'

section "Rag-proxy query (source filter)"
resp=$(curl_timed "http://localhost:8080/query" \
  "{\"q\":\"2025 개정 세법 요약\",\"k\":5,\"source\":\"${FILE_SRC}\"}")
echo "$resp" | jq -r '
  "hits=\(.hits)",
  "scores=" + ([(.items[]?.score // .contexts[]?.score // .items[]?.distance // .contexts[]?.distance)] | tostring),
  "notes=" + ((.notes // {}) | tostring)
'

section "Rag-router metadata (file hint)"
resp=$(curl_timed "http://localhost:8088/v1/chat/completions" "{
  \"model\":\"/model/Qwen2.5-14B-Instruct\",
  \"messages\":[{\"role\":\"user\",\"content\":\"2025 개정 세법 요약\"}],
  \"metadata\":{\"files\":[{\"rag_source\":\"${FILE_SRC}\"}]}
}")
echo "$resp" | jq '.choices[0].message.content,.trace_id'

section "Rag-router forced Confluence (keyword)"
resp=$(curl_timed "http://localhost:8088/v1/chat/completions" "{
  \"model\":\"/model/Qwen2.5-14B-Instruct\",
  \"messages\":[{\"role\":\"user\",\"content\":\"컨플루언스에서 배홍진 일일업무보고 최근 7일 요약\"}]
}")
echo "$resp" | jq '.choices[0].message.content,.trace_id'

section "Rag-router forced Confluence (pageId)"
resp=$(curl_timed "http://localhost:8088/v1/chat/completions" "{
  \"model\":\"/model/Qwen2.5-14B-Instruct\",
  \"messages\":[{\"role\":\"user\",\"content\":\"pageId=${PAGE_ID} 내용 요약\"}]
}")
echo "$resp" | jq '.choices[0].message.content,.trace_id'

section "NO_RAG sanity"
resp=$(curl_timed "http://localhost:8088/v1/chat/completions" "{
  \"model\":\"/model/Qwen2.5-14B-Instruct\",
  \"messages\":[{\"role\":\"user\",\"content\":\"메갈로돈 설명해줘\"}]
}")
echo "$resp" | jq '.choices[0].message.content,.trace_id'

section "Logs (last 10m)"
docker logs --since 10m rag-router | grep -E "trace_id=|route_state|file_hint_src|meta_sources" || true
docker logs --since 10m rag-proxy  | grep -E "trace_id=|src_filter|MCP" || true

section "Rag-router Docker focus"
time curl -s http://localhost:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"/model/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"리눅스 도커 세미나 도커 내용 위주로 자세히 요약해줘"}]}' \
  | jq '.choices[0].message.content'

section "Rag-router AI docx masking"
time curl -s http://localhost:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"/model/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"AI 관련 작업 내역 요약"}]}' \
  | jq '.choices[0].message.content'
