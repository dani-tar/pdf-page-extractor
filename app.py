from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form, Query, Body
import asyncpg
import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Dict, Any

import boto3
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig

import pypdfium2 as pdfium


app = FastAPI()

API_KEY = os.environ.get("API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Cloudflare R2 (S3-compatible)
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")  # optional override

DEFAULT_BATCH_PAGES = int(os.environ.get("BATCH_PAGES", "50"))
PROGRESS_EVERY_PAGES = int(os.environ.get("PROGRESS_EVERY_PAGES", "10"))

_pool: asyncpg.Pool | None = None
_s3 = None


# ----------------------------
# DB helpers
# ----------------------------
async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    return _pool


async def db_one(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(sql, *args)


async def db_exec(sql: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)


def require_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ----------------------------
# R2 client
# ----------------------------
def get_r2_endpoint() -> str:
    if R2_ENDPOINT_URL:
        return R2_ENDPOINT_URL
    if not R2_ACCOUNT_ID:
        raise RuntimeError("R2_ACCOUNT_ID is not set")
    return f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"


def get_s3():
    global _s3
    if _s3 is None:
        if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME):
            raise RuntimeError("R2 credentials missing (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET_NAME)")
        _s3 = boto3.client(
            "s3",
            endpoint_url=get_r2_endpoint(),
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3


TRANSFER_CFG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=2,
    use_threads=True,
)


# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------
# Generic helpers
# ----------------------------
async def save_upload_to_tmp(upload: UploadFile) -> str:
    tmp_dir = Path(tempfile.gettempdir())
    fd, tmp_path = tempfile.mkstemp(prefix="pdf_", suffix=".pdf", dir=str(tmp_dir))
    os.close(fd)

    upload.file.seek(0)
    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    return tmp_path


def make_r2_key(document_id: str, changed_id: str, job_id: uuid.UUID) -> str:
    safe_changed = changed_id.replace("/", "_")
    return f"pdf/{document_id}/{safe_changed}/{job_id}.pdf"


def upload_file_to_r2(local_path: str, key: str):
    s3 = get_s3()
    s3.upload_file(local_path, R2_BUCKET_NAME, key, Config=TRANSFER_CFG)


def download_file_from_r2(key: str, local_path: str):
    s3 = get_s3()
    s3.download_file(R2_BUCKET_NAME, key, local_path)


async def set_error(job_id: uuid.UUID, message: str):
    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET status='failed',
            last_error=$2,
            error=$2,
            updated_at=now(),
            completed_at=now()
        WHERE job_id=$1
        """,
        job_id, message
    )


# ----------------------------
# R2: list + delete by prefix (cleanup workflow)
# ----------------------------
def list_r2_keys_all(prefix: str, page_size: int = 1000) -> List[str]:
    """
    Lists ALL keys for a prefix, using ContinuationToken pagination.
    """
    s3 = get_s3()
    keys: List[str] = []
    token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": R2_BUCKET_NAME,
            "Prefix": prefix,
            "MaxKeys": page_size,
        }
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", []) or []
        for obj in contents:
            k = obj.get("Key")
            if k:
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            if not token:
                break
        else:
            break

    return keys


def delete_r2_keys_batched(keys: List[str]) -> Dict[str, Any]:
    """
    Deletes keys in batches of 1000 (S3 API limit).
    Returns a robust summary:
      - deleted: number of keys reported deleted by API
      - errors: list of errors returned by API
      - requested: number of keys requested to delete
    """
    s3 = get_s3()
    deleted_total = 0
    errors_total: List[Dict[str, Any]] = []

    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        resp = s3.delete_objects(
            Bucket=R2_BUCKET_NAME,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted_total += len(resp.get("Deleted", []) or [])
        errors_total.extend(resp.get("Errors", []) or [])

    return {"requested": len(keys), "deleted": deleted_total, "errors": errors_total}


@app.get("/r2/list")
async def r2_list(prefix: str, x_api_key: str = Header(None)):
    require_api_key(x_api_key)

    prefix = (prefix or "").strip()
    if not prefix:
        raise HTTPException(status_code=422, detail="prefix is required")

    try:
        keys = await asyncio.to_thread(list_r2_keys_all, prefix)
        return {
            "prefix": prefix,
            "count": len(keys),
            "keys": keys[:200],  # cap list for response size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"R2 list failed: {str(e)}")


@app.post("/r2/delete-prefix")
async def r2_delete_prefix(payload: dict = Body(...), x_api_key: str = Header(None)):
    require_api_key(x_api_key)

    prefix = (payload or {}).get("prefix")
    if not prefix or not isinstance(prefix, str):
        raise HTTPException(status_code=422, detail="payload must include string field 'prefix'")

    prefix = prefix.strip()

    # Safety guard: only allow deleting under pdf/
    if not prefix.startswith("pdf/"):
        raise HTTPException(status_code=400, detail="Refusing to delete: prefix must start with 'pdf/'")

    try:
        keys = await asyncio.to_thread(list_r2_keys_all, prefix)
        if not keys:
            return {"prefix": prefix, "listed": 0, "deleted": 0, "errors": 0}

        result = await asyncio.to_thread(delete_r2_keys_batched, keys)

        # If partial errors, fail hard so n8n sees it
        if result["errors"]:
            raise RuntimeError(f"Partial delete: {result}")

        return {
            "prefix": prefix,
            "listed": len(keys),
            "deleted": result["deleted"],
            "errors": 0,
            "sample": keys[:3],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"R2 delete-prefix failed: {str(e)}")


# ----------------------------
# PDF Extract API
# ----------------------------
@app.post("/extract/start")
async def extract_start(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    changed_id: str = Form(...),
    file_id: str | None = Form(None),
    title: str | None = Form(None),
    source: str | None = Form(None),
    url: str | None = Form(None),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)

    try:
        doc_uuid = uuid.UUID(document_id)
    except Exception:
        raise HTTPException(status_code=422, detail="document_id must be a UUID string")

    existing = await db_one(
        """
        SELECT job_id, status, r2_key
        FROM public.pdf_extract_jobs
        WHERE document_id=$1 AND changed_id=$2
        ORDER BY created_at DESC
        LIMIT 1
        """,
        doc_uuid, changed_id
    )

    if existing:
        job_id = existing["job_id"]
        await db_exec(
            """
            UPDATE public.pdf_extract_jobs
            SET
              file_id = COALESCE(NULLIF($2,''), file_id),
              title   = COALESCE(NULLIF($3,''), title),
              source  = COALESCE(NULLIF($4,''), source),
              url     = COALESCE(NULLIF($5,''), url),
              updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            file_id or "",
            title or "",
            source or "",
            url or "",
        )

        # If r2_key exists, skip re-upload
        if existing["r2_key"]:
            row = await db_one(
                """
                SELECT job_id, status, num_pages, inserted_pages, next_page, last_error, error, r2_key
                FROM public.pdf_extract_jobs
                WHERE job_id=$1
                """,
                job_id
            )
            return {
                "job_id": str(row["job_id"]),
                "status": row["status"],
                "num_pages": row["num_pages"],
                "inserted_pages": row["inserted_pages"],
                "next_page": row["next_page"],
                "last_error": row["last_error"],
                "error": row["error"],
                "r2_key": row["r2_key"],
                "idempotent": True,
                "uploaded": False,
            }
    else:
        job_id = uuid.uuid4()
        await db_exec(
            """
            INSERT INTO public.pdf_extract_jobs
              (job_id, document_id, changed_id, source, file_id, title, url, status, created_at, updated_at, inserted_pages, next_page)
            VALUES
              ($1, $2, $3, $4, $5, $6, $7, 'queued', now(), now(), 0, 1)
            """,
            job_id, doc_uuid, changed_id, source, file_id, title, url
        )

    tmp_path = await save_upload_to_tmp(file)
    r2_key = make_r2_key(document_id, changed_id, job_id)

    try:
        await asyncio.to_thread(upload_file_to_r2, tmp_path, r2_key)
    except Exception as e:
        await set_error(job_id, f"R2 upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    await db_exec(
        """
        UPDATE public.pdf_extract_jobs
        SET r2_key=$2,
            next_page=COALESCE(next_page, 1),
            last_error=NULL,
            error=NULL,
            updated_at=now()
        WHERE job_id=$1
        """,
        job_id, r2_key
    )

    return {"job_id": str(job_id), "status": "queued", "r2_key_saved": True, "uploaded": True}


@app.get("/extract/status/{job_id}")
async def extract_status(job_id: str, x_api_key: str = Header(None)):
    require_api_key(x_api_key)

    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    row = await db_one(
        """
        SELECT job_id, document_id, changed_id, status, num_pages, inserted_pages, next_page,
               last_error, error, created_at, updated_at, started_at, completed_at,
               source, file_id, title, url, r2_key
        FROM public.pdf_extract_jobs
        WHERE job_id=$1
        """,
        job_uuid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    def iso(v):
        return v.isoformat() if v else None

    return {
        "job_id": str(row["job_id"]),
        "document_id": str(row["document_id"]),
        "changed_id": row["changed_id"],
        "status": row["status"],
        "num_pages": row["num_pages"],
        "inserted_pages": row["inserted_pages"],
        "next_page": row["next_page"],
        "last_error": row["last_error"],
        "error": row["error"],
        "source": row["source"],
        "file_id": row["file_id"],
        "title": row["title"],
        "url": row["url"],
        "r2_key": row["r2_key"],
        "created_at": iso(row["created_at"]),
        "updated_at": iso(row["updated_at"]),
        "started_at": iso(row["started_at"]),
        "completed_at": iso(row["completed_at"]),
    }


@app.get("/extract/pages/{job_id}")
async def extract_pages(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)

    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    job = await db_one(
        "SELECT status, num_pages, inserted_pages, last_error, error FROM public.pdf_extract_jobs WHERE job_id=$1",
        job_uuid
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
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
        "last_error": job["last_error"],
        "error": job["error"],
        "offset": offset,
        "limit": limit,
        "pages": [{"pdf_page": r["pdf_page"], "text": r["text"]} for r in rows],
    }


@app.post("/extract/work/{job_id}")
async def extract_work(
    job_id: str,
    max_pages: int = Query(DEFAULT_BATCH_PAGES, ge=1, le=500),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)

    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=422, detail="job_id must be a UUID string")

    pool = await get_pool()
    async with pool.acquire() as conn:
        # per-job advisory lock for the full /work call
        locked = await conn.fetchval("SELECT pg_try_advisory_lock(hashtext($1))", str(job_uuid))
        if not locked:
            raise HTTPException(status_code=409, detail="Job is currently being processed (lock busy)")

        pdf_path = None
        try:
            job = await conn.fetchrow(
                """
                SELECT job_id, status, num_pages, inserted_pages, last_error, error, r2_key, next_page
                FROM public.pdf_extract_jobs
                WHERE job_id=$1
                """,
                job_uuid
            )
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            if not job["r2_key"]:
                raise HTTPException(status_code=400, detail="Job has no r2_key (did /extract/start complete?)")

            if job["status"] == "done":
                return {
                    "job_id": job_id,
                    "status": "done",
                    "num_pages": job["num_pages"],
                    "inserted_pages": job["inserted_pages"],
                    "next_page": job["next_page"],
                }

            if job["status"] in ("queued", "failed"):
                await conn.execute(
                    """
                    UPDATE public.pdf_extract_jobs
                    SET status='running',
                        started_at=COALESCE(started_at, now()),
                        updated_at=now(),
                        last_error=NULL,
                        error=NULL
                    WHERE job_id=$1
                    """,
                    job_uuid
                )

            # download PDF
            tmp_dir = Path(tempfile.gettempdir())
            fd, pdf_path = tempfile.mkstemp(prefix=f"job_{job_id}_", suffix=".pdf", dir=str(tmp_dir))
            os.close(fd)

            await asyncio.to_thread(download_file_from_r2, job["r2_key"], pdf_path)

            pdf = pdfium.PdfDocument(pdf_path)
            try:
                num_pages = len(pdf)
            except Exception:
                await set_error(job_uuid, "Failed to read PDF page count")
                raise HTTPException(status_code=500, detail="Failed to read PDF page count")

            # persist num_pages if missing
            await conn.execute(
                """
                UPDATE public.pdf_extract_jobs
                SET num_pages = COALESCE(num_pages, $2),
                    updated_at = now()
                WHERE job_id=$1
                """,
                job_uuid, num_pages
            )

            start_page = int(job["next_page"] or 1)
            if start_page < 1:
                start_page = 1

            if start_page > num_pages:
                await conn.execute(
                    """
                    UPDATE public.pdf_extract_jobs
                    SET status='done',
                        completed_at=COALESCE(completed_at, now()),
                        updated_at=now()
                    WHERE job_id=$1
                    """,
                    job_uuid
                )
                return {
                    "job_id": job_id,
                    "status": "done",
                    "num_pages": num_pages,
                    "inserted_pages": int(job["inserted_pages"] or 0),
                    "next_page": start_page,
                    "batch_start": start_page,
                    "batch_end": start_page - 1,
                    "batch_processed": 0,
                    "batch_inserted": 0,
                }

            end_page = min(start_page + int(max_pages) - 1, num_pages)

            processed = 0
            inserted_in_call = 0

            for page_no in range(start_page, end_page + 1):
                page = pdf.get_page(page_no - 1)
                textpage = page.get_textpage()
                text = (textpage.get_text_range() or "")
                textpage.close()
                page.close()

                r = await conn.fetchrow(
                    """
                    INSERT INTO public.pdf_extract_pages (job_id, pdf_page, text)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (job_id, pdf_page) DO NOTHING
                    RETURNING 1 AS inserted
                    """,
                    job_uuid, page_no, text
                )
                if r:
                    inserted_in_call += 1

                processed += 1

                if processed % 10 == 0:
                    await asyncio.sleep(0)

                if processed % PROGRESS_EVERY_PAGES == 0 and inserted_in_call:
                    await conn.execute(
                        """
                        UPDATE public.pdf_extract_jobs
                        SET inserted_pages = inserted_pages + $2,
                            updated_at = now()
                        WHERE job_id=$1
                        """,
                        job_uuid, inserted_in_call
                    )
                    inserted_in_call = 0

            if inserted_in_call:
                await conn.execute(
                    """
                    UPDATE public.pdf_extract_jobs
                    SET inserted_pages = inserted_pages + $2,
                        updated_at = now()
                    WHERE job_id=$1
                    """,
                    job_uuid, inserted_in_call
                )

            final = await conn.fetchrow(
                "SELECT inserted_pages, num_pages FROM public.pdf_extract_jobs WHERE job_id=$1",
                job_uuid
            )
            final_inserted = int(final["inserted_pages"] or 0)
            final_num_pages = int(final["num_pages"] or num_pages)

            new_next_page = end_page + 1
            status = "done" if final_inserted >= final_num_pages else "running"

            await conn.execute(
                """
                UPDATE public.pdf_extract_jobs
                SET next_page=$2,
                    status=$3,
                    completed_at=CASE WHEN $3='done' THEN now() ELSE completed_at END,
                    updated_at=now(),
                    last_error=NULL,
                    error=NULL
                WHERE job_id=$1
                """,
                job_uuid, new_next_page, status
            )

            return {
                "job_id": job_id,
                "status": status,
                "num_pages": final_num_pages,
                "inserted_pages": final_inserted,
                "batch_start": start_page,
                "batch_end": end_page,
                "batch_processed": processed,
                "next_page": new_next_page,
            }

        except HTTPException:
            raise
        except Exception as e:
            await set_error(job_uuid, str(e))
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # cleanup temp file
            if pdf_path:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
            # release advisory lock
            try:
                await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", str(job_uuid))
            except Exception:
                pass
