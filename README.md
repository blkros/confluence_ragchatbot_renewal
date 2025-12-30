# confluence_ragchatbot_renewal

## Upload Support (RAG)
- PDF, PPTX, XLSX, CSV, TXT, LOG, MD
- PPTX: slide text + table extraction + speaker notes; optional image OCR
- XLSX: header-aware row extraction with sheet metadata

## Optional Env
- `PPTX_OCR=1` to OCR images in slides (requires Tesseract)
- `XLSX_MAX_ROWS_PER_SHEET` (default: 2000)
- `XLSX_MAX_COLS` (default: 50)
