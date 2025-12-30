# confluence_ragchatbot_renewal

## Upload Support (RAG)
- PDF, DOCX, PPTX, XLSX, CSV, TXT, LOG, MD
- PPTX: slide text + table extraction + speaker notes; optional image OCR
- XLSX: header-aware row extraction with sheet metadata
- DOCX: paragraph + table extraction

## Optional Env
- `PPTX_OCR=1` to OCR images in slides (requires Tesseract)
- `XLSX_MAX_ROWS_PER_SHEET` (default: 2000)
- `XLSX_MAX_COLS` (default: 50)
- `ROUTER_INGEST_WAIT_SEC` (default: 8)
- `ROUTER_INGEST_WAIT_INTERVAL` (default: 1.0)
- `ROUTER_KO_MORPH=1` to enable Korean morphological tokenization (requires kiwipiepy)
- `ROUTER_UPLOADS_DIR` (default: /data/uploads)
