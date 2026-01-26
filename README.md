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

## Problem
We needed a faster way to search internal Confluence pages and uploaded files.

## Constraint
SSO/access control frequently blocked REST API calls; WebUI MCP integration was not sufficient.

## Solution
Implemented a RAG pipeline with a router/proxy separation and an SSO-safe Confluence fallback.

## Status
PoC verified on an on-prem GPU server; preparing for production hardening.

## Architecture
![RAG architecture](docs/assets/포트폴리오.jpg)
- open-webui: UI
- rag-router: LLM direct vs RAG route + masking/strict gate
- rag-proxy: FAISS retrieval + rerank + context build, fallback to mcp-confluence
