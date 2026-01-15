from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import pdfplumber
import io
import os

app = FastAPI()
API_KEY = os.environ.get("API_KEY")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # Simple API key auth (recommended)
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    pdf_bytes = await file.read()
    pages = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"pdf_page": i, "text": text})

    return {"numpages": len(pages), "pages": pages}
