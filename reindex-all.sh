#!/bin/bash
# 모든 파일을 재인덱싱 (overwrite 모드)

UPLOAD_DIR="storage/uploads"
RAG_URL="http://localhost:8080"

echo "Starting bulk reindex from $UPLOAD_DIR..."

# find로 파일 찾기
count=0
total=$(find "$UPLOAD_DIR" -maxdepth 1 -type f \( -name "*.pdf" -o -name "*.pptx" -o -name "*.docx" -o -name "*.xlsx" \) | wc -l)

echo "Found $total files to index"
echo ""

find "$UPLOAD_DIR" -maxdepth 1 -type f \( -name "*.pdf" -o -name "*.pptx" -o -name "*.docx" -o -name "*.xlsx" \) | sort | while read -r file; do
  ((count++))
  basename=$(basename "$file")
  echo "[$count/$total] Indexing: $basename"
  
  result=$(curl -s -X POST "$RAG_URL/ingest" \
    -F "file=@$file" \
    -F "overwrite=true")
  
  indexed=$(echo "$result" | jq -r '.indexed // empty')
  error=$(echo "$result" | jq -r '.error // .detail // empty')
  
  if [ -n "$indexed" ]; then
    echo "  ✓ Indexed $indexed chunks"
  elif [ -n "$error" ]; then
    echo "  ✗ Error: $error"
  else
    echo "  ? Unknown response"
  fi
  
  sleep 0.2
done

echo ""
echo "Done! Final index stats:"
curl -s "$RAG_URL/index/stats" | jq '{doc_total, sources: (.sources | length)}'
