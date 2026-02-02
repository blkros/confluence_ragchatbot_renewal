#!/bin/bash
# 모든 파일을 재인덱싱 (overwrite 모드)

UPLOAD_DIR="storage/uploads"
RAG_URL="http://localhost:8080"

echo "Starting bulk reindex from $UPLOAD_DIR..."

count=0
total=$(find "$UPLOAD_DIR" -type f \( -name "*.pdf" -o -name "*.pptx" -o -name "*.docx" -o -name "*.xlsx" \) 2>/dev/null | wc -l)

for file in "$UPLOAD_DIR"/*.{pdf,pptx,docx,xlsx} 2>/dev/null; do
  [ -f "$file" ] || continue
  
  ((count++))
  basename=$(basename "$file")
  echo "[$count/$total] Indexing: $basename"
  
  curl -s -X POST "$RAG_URL/ingest" \
    -F "file=@$file" \
    -F "overwrite=true" \
    | jq -r '.indexed // .error // "unknown"'
  
  sleep 0.3
done

echo ""
echo "Done! Final index stats:"
curl -s "$RAG_URL/index/stats" | jq '{doc_total, sources: (.sources | length)}'
