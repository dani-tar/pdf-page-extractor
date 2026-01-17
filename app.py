from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import pdfplumber
import io
import os

app = FastAPI()
API_KEY = os.environ.get("API_KEY")

@app.get("/health")
def health():
    return {"ok": True}

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.background import BackgroundTasks
import tempfile, os, uuid, datetime

app = FastAPI()

# TODO: bytt til din DB-klient (psycopg/asyncpg/sqlalchemy)
def db_exec(query: str, params: dict = None):
    raise NotImplementedError

def db_one(query: str, params: dict = None):
    raise NotImplementedError

def extract_pages_to_text(pdf_path: str):
    """
    Generator som yield'er (page_number, text) for hver side.
    Bruk din eksisterende PDF-extract her (pdfplumber/pypdf/fitz).
    """
    raise NotImplementedError

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/extract/start")
def extract_start(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    changed_id: str = Form(...),
):
    # 1) valider
    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=422, detail="document_id must be UUID")

    # 2) idempotent jobb-opprettelse: én jobb per (document_id, changed_id)
    job = db_one("""
        SELECT job_id, status, total_pages, processed_pages
        FROM pdf_extract_jobs
        WHERE document_id = %(document_id)s AND changed_id = %(changed_id)s
        LIMIT 1
    """, {"document_id": str(doc_uuid), "changed_id": changed_id})

    if job:
        job_id = job["job_id"]
        # hvis allerede ferdig: returner bare jobben
        if job["status"] == "completed":
            return {"job_id": job_id, "status": "completed", "total_pages": job["total_pages"]}
    else:
        job_id = str(uuid.uuid4())
        db_exec("""
            INSERT INTO pdf_extract_jobs (job_id, document_id, changed_id, status, processed_pages, updated_at)
            VALUES (%(job_id)s, %(document_id)s, %(changed_id)s, 'queued', 0, now())
        """, {"job_id": job_id, "document_id": str(doc_uuid), "changed_id": changed_id})

    # 3) lagre fil til temp (eller GCS/S3 hvis du vil være helt “enterprise”)
    suffix = os.path.splitext(file.filename or "")[-1] or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(file.file.read())

    # 4) start bakgrunnsprosess (resumable)
    background.add_task(process_job, job_id, tmp_path)

    return {"job_id": job_id, "status": "running_or_queued"}

def process_job(job_id: str, tmp_path: str):
    try:
        # mark running
        db_exec("""
            UPDATE pdf_extract_jobs
            SET status='running', started_at=COALESCE(started_at, now()), updated_at=now()
            WHERE job_id=%(job_id)s
        """, {"job_id": job_id})

        # finn sist lagrede side (for resume)
        row = db_one("""
            SELECT COALESCE(MAX(page_number), 0) AS last_page
            FROM pdf_extract_pages
            WHERE job_id=%(job_id)s
        """, {"job_id": job_id})
        last_page = int(row["last_page"] or 0)

        processed = last_page

        # TODO: sett total_pages hvis du kan beregne det tidlig
        # db_exec("UPDATE pdf_extract_jobs SET total_pages=%(n)s WHERE job_id=%(job_id)s", ...)

        for page_number, text in extract_pages_to_text(tmp_path):
            if page_number <= last_page:
                continue

            db_exec("""
                INSERT INTO pdf_extract_pages (job_id, page_number, text)
                VALUES (%(job_id)s, %(page_number)s, %(text)s)
                ON CONFLICT (job_id, page_number) DO UPDATE
                SET text = EXCLUDED.text
            """, {"job_id": job_id, "page_number": page_number, "text": text})

            processed = page_number

            db_exec("""
                UPDATE pdf_extract_jobs
                SET processed_pages=%(processed)s, updated_at=now()
                WHERE job_id=%(job_id)s
            """, {"job_id": job_id, "processed": processed})

        db_exec("""
            UPDATE pdf_extract_jobs
            SET status='completed', completed_at=now(), updated_at=now()
            WHERE job_id=%(job_id)s
        """, {"job_id": job_id})

    except Exception as e:
        db_exec("""
            UPDATE pdf_extract_jobs
            SET status='failed', last_error=%(err)s, updated_at=now()
            WHERE job_id=%(job_id)s
        """, {"job_id": job_id, "err": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.get("/extract/status")
def extract_status(job_id: str):
    job = db_one("""
        SELECT job_id, status, total_pages, processed_pages, updated_at, last_error
        FROM pdf_extract_jobs
        WHERE job_id=%(job_id)s
    """, {"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.get("/extract/pages")
def extract_pages(job_id: str, limit: int = 50, after_page: int = 0):
    rows = db_exec("""
        SELECT page_number, text
        FROM pdf_extract_pages
        WHERE job_id=%(job_id)s AND page_number > %(after_page)s
        ORDER BY page_number
        LIMIT %(limit)s
    """, {"job_id": job_id, "after_page": after_page, "limit": limit})
    # db_exec bør returnere liste av dicts her
    return {"job_id": job_id, "pages": rows}

