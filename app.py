from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse
import pdfplumber
import io
import os
import uuid
import asyncio
import asyncpg
from datetime import datetime, timezone

app = FastAPI()

API_KEY = os.environ.get("API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")  # Neon connection string, e.g. postgres://...

# ----------------------------
# DB helpers
# ----------------------------

_pool: asyncpg.Pool | None = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _pool

async def db_exec(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)

async def db_one(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(sql, *args)

async def db_all(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *args)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ----------------------------
# Schema bootstrap (safe)
# ----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.pdf_extract_jobs (
  job_id uuid PRIMARY KEY,
  document_id uuid,
  changed_id text,
  file_id text,
  title text,
  source text,
  url text,
  status text NOT NULL DEFAULT 'queued', -- queued|running|done|failed
  num_pages integer,
  inserted_pages integer NOT NULL DEFAULT 0,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  completed_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS pdf_extract_jobs_doc_changed_uq
ON public.pdf_extract_jobs (document_id, changed_id);

CREATE TABLE IF NOT EXISTS public.pdf_extract_pages (
  job_id uuid NOT NULL REFERENCES public.pdf_extract_jobs(job_id) ON DELETE CASCADE,
  pdf_page integer NOT NULL,
  text text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, pdf_page)
);

CREATE INDEX IF NOT EXISTS pdf_extract_pages_job_page_idx
ON public.pdf_extract_pages (job_id, pdf_page);
"""

@app.on_event("startup")
async def _startup():
    # Create pool + ensure tables exist
    await get_pool()
    # asyncpg doesn't support multiple statements in one execute reliably depending on settings.
    # We'll split on semicolons safely for this small bootstrap.
    statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]
    for stmt in statements:
        await db_exec(stmt)

# ----------------------------
# Auth
# ----------------------------

def require_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ----------------------------
# Health
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------
# B2: Job-based extraction
# ----------------------------

@app.post("/extract/start")
async def extract_start(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    changed_id: str = Form(...),
    file_id: str | None = Form(None),
    title: str | None = Form(None),
    source: str | None = Form("google_drive"),
    url: str | None = Form(None),
    x_api_key: str = Header(None),
):
    """
    Starts a PDF extraction job and returns quickly with job_id.
    Stores per-page text into Neon (pdf_extract_pages).
    """
    require_api_key(x_api_key)

    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=422, detail="document_id must be a UUID string")

    # Idempotency: if the same (document_id, changed_id) already has a job that isn't failed,
    # return it (so retries from n8n don't spawn duplicates).
    existing = await db_one(
        """
        SELECT job_id, status, num_pages, inserted_pages, error
        FROM public.pdf_extract_jobs
        WHERE document_id=$1 AND changed_id=$2
        """,
        doc_uuid, changed_id
    )
    if existing and existing["status"] in ("queued", "running", "done"):
        return {
            "job_id": str(existing["job_id"]),
            "status": existing["status"],
            "num_pages": existing["num_pages"],
            "inserted_pages": existing["inserted_pages"],
            "error": existing["error"],
            "idempotent": True,
        }

    job_id = uuid.uuid4()

    # Read file bytes now (request can close; background task needs bytes)
    pdf_bytes = await file.read()

    await db_exec(
        """
        INSERT INTO public.pdf_extract_jobs
          (job_id, document_id, changed_id, file_id, title, source, url, status, created_at, updated_at)
        VALUES
          ($1, $2, $3, $4, $5, $6, $7, 'queued', now(), now())
        ON CONFLICT (document_id, changed_id)
        DO UPDATE SET
          updated_at=now()
        """,
        job_id, doc_uuid, changed_id, file_id, title, source, url
    )

    background_tasks.add_task(process_job, job_id, pdf_bytes)

    return {"job_id": str(job_id), "status": "queued"}

@app.get("/extract/status/{job_id}")
async def extract_status(job_id: str, x_api_key: str = Header(None)):
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    row = await db_one(
        """
        SELECT job_id, document_id, changed_id, status, num_pages, inserted_pages, error, created_at, updated_at, started_at, completed_at
        FROM public.pdf_extract_jobs
        WHERE job_id=$1
        """,
        job_uuid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": str(row["job_id"]),
        "document_id": str(row["document_id"]) if row["document_id"] else None,
        "changed_id": row["changed_id"],
        "status": row["status"],
        "num_pages": row["num_pages"],
        "inserted_pages": row["inserted_pages"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }

@app.get("/extract/pages/{job_id}")
async def extract_pages(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    x_api_key: str = Header(None),
):
    """
    Returns pages already extracted for a job, in a stable order.
    Use offset/limit for batching (n8n-friendly).
    """
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    # Validate job exists
    job = await db_one(
        "SELECT status, num_pages, inserted_pages, error FROM public.pdf_extract_jobs WHERE job_id=$1",
        job_uuid
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    rows = await db_all(
        """
        SELECT pdf_page, text
        FROM public.pdf_extract_pages
        WHERE job_id=$1
        ORDER BY pdf_page
        OFFSET $2 LIMIT $3
        """,
        job_uuid, offset, limit
    )

    return {
        "job_id": job_id,
        "status": job["status"],
        "num_pages": job["num_pages"],
        "inserted_pages": job["inserted_pages"],
        "error": job["error"],
        "offset": offset,
        "limit": limit,
        "pages": [{"pdf_page": r["pdf_page"], "text": r["text"]} for r in rows],
    }

@app.post("/extract/retry/{job_id}")
async def extract_retry(job_id: str, x_api_key: str = Header(None)):
    """
    Optional helper: mark a failed job as queued again (pages already stored remain).
    You can call this from n8n if you detect a failed/stale run.
    """
    require_api_key(x_api_key)
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    row = await db_one("SELECT status FROM public.pdf_extract_jobs WHERE job_id=$1", job_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] not in ("failed",):
        return {"job_id": job_id, "status": row["status"], "note": "Job not in failed state"}

    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET status='queued', error=NULL, updated_at=now(), started_at=NULL, completed_at=NULL
        WHERE job_id=$1
        """,
        job_uuid
    )
    return {"job_id": job_id, "status": "queued"}

# ----------------------------
# Worker: process one job
# ----------------------------

async def process_job(job_id: uuid.UUID, pdf_bytes: bytes):
    # Mark running
    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET status='running', started_at=now(), updated_at=now(), error=NULL
        WHERE job_id=$1
        """,
        job_id
    )

    try:
        pages_out = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
            await db_exec(
                """
                UPDATE public.pdf_extract_jobs
                SET num_pages=$2, updated_at=now()
                WHERE job_id=$1
                """,
                job_id, num_pages
            )

            # Extract + upsert per page (so we can resume / partial-read)
            inserted = 0
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Upsert each page
                await db_exec(
                    """
                    INSERT INTO public.pdf_extract_pages (job_id, pdf_page, text)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (job_id, pdf_page) DO UPDATE SET text=EXCLUDED.text
                    """,
                    job_id, i, text
                )
                inserted += 1

                # Update progress every N pages (reduce DB chatter a bit)
                if inserted % 10 == 0 or inserted == num_pages:
                    await db_exec(
                        """
                        UPDATE public.pdf_extract_jobs
                        SET inserted_pages=$2, updated_at=now()
                        WHERE job_id=$1
                        """,
                        job_id, inserted
                    )

                # Yield to event loop occasionally (keeps server responsive)
                if inserted % 5 == 0:
                    await asyncio.sleep(0)

        # Done
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='done', inserted_pages=COALESCE(num_pages, inserted_pages), completed_at=now(), updated_at=now()
            WHERE job_id=$1
            """,
            job_id
        )

    except Exception as e:
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET status='failed', error=$2, updated_at=now(), completed_at=now()
            WHERE job_id=$1
            """,
            job_id, str(e)
        )
