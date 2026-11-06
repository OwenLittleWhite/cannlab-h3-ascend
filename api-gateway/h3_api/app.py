import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import os
from pathlib import Path
import uuid
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx

from .compiler import canonical, compile_request, digest
from .guard import AscendProtection, ProtectionError
from .schema import GenerateRequest
from .store import Conflict, JobStore


@dataclass
class Settings:
    profile: dict
    state_dir: Path
    token: str


def combo_options(spec):
    # ComfyUI v1 exposes [choices, metadata]; v3 exposes ['COMBO', {options: choices}].
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if len(spec)>1 and isinstance(spec[1], dict) and isinstance(spec[1].get('options'), list):
        return [item['key'] if isinstance(item, dict) and 'key' in item else item for item in spec[1]['options']]
    return None


def make_app(settings: Settings, *, guard=None, client=None):
    settings.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = JobStore(settings.state_dir / "jobs.sqlite3")
    policy_class = AscendProtection
    protection = guard if guard is not None else policy_class(settings.profile)
    http = client or httpx.AsyncClient(timeout=30, trust_env=False)
    submit_lock = asyncio.Lock()
    profile = settings.profile
    proxy = profile["proxy_url"].rstrip("/")
    auth = {"Authorization": "Bearer " + settings.token}

    @asynccontextmanager
    async def lifespan(app):
        yield
        if client is None:
            await http.aclose()

    app = FastAPI(title="H3 Workflow API", version="0.2.0", lifespan=lifespan,
                  description=(Path(__file__).resolve().parent.parent / "API_REFERENCE.md").read_text())
    app.state.store = store
    bearer = HTTPBearer(auto_error=False)

    async def authorized(credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not credential or credential.scheme.lower() != "bearer" or not hmac.compare_digest(credential.credentials, settings.token):
            raise HTTPException(401, "A valid Bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    def protected():
        try:
            return protection.check()
        except ProtectionError as error:
            raise HTTPException(503, {"code": "protected_backend_unavailable", "message": str(error)}) from error

    async def upstream(method, path, **kwargs):
        try:
            response = await http.request(method, proxy + path, headers=auth | kwargs.pop("headers", {}), **kwargs)
        except httpx.HTTPError as error:
            raise HTTPException(503, {"code": "proxy_unavailable", "message": str(error)}) from error
        return response

    def rewrite(value, request):
        base = str(request.base_url).rstrip("/")
        if isinstance(value, str) and value.startswith(proxy+"/"):
            return base+value[len(proxy):]
        if isinstance(value, dict):
            return {k: rewrite(v, request) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v, request) for v in value]
        return value

    def public(row):
        compiled = json.loads(row["compiled"])
        return {"id": row["id"], "status": row["state"], "proxy_job_id": row["proxy_id"], "created_at_unix": row["created"],
                "applied_parameters": compiled["applied_parameters"], "final_prompt": compiled["final_prompt"],
                "workflow_sha256": compiled["workflow_sha256"], "prompt_sha256": compiled["prompt_sha256"],
                "warnings": compiled["warnings"], "error": row["error"],
                "links": {"self": "/v1/jobs/"+row["id"], "workflow": "/v1/jobs/"+row["id"]+"/workflow", "events": "/v1/jobs/"+row["id"]+"/events"}}

    async def validate_assets(compiled):
        for item in compiled["inputs"]:
            response = await upstream("GET", "/api/v2/assets/"+quote(item["asset_id"], safe=""))
            if response.status_code != 200:
                raise HTTPException(422, {"code": "missing_asset", "asset_id": item["asset_id"], "meaning": item["meaning"]})
            content_type = response.json().get("content_type", "")
            if not content_type.startswith(item["kind"]+"/"):
                raise HTTPException(422, {"code": "asset_type_mismatch", "meaning": item["meaning"], "expected": item["kind"], "actual": content_type})

    async def check_nodes(compiled):
        try:
            response = await http.get(profile["comfy_url"].rstrip("/")+"/object_info")
            response.raise_for_status()
            info = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, {"code": "comfy_unavailable", "message": str(error)}) from error
        missing = sorted({node["class_type"] for node in compiled["workflow"].values()} - info.keys())
        if missing:
            raise HTTPException(503, {"code": "missing_nodes", "nodes": missing, "message": "Install the mapping adapter and load it when the protected backend is idle"})
        for node in compiled["workflow"].values():
            schema = info[node["class_type"]].get("input", {})
            choices = schema.get("required", {}) | schema.get("optional", {})
            for field, value in node["inputs"].items():
                spec = choices.get(field)
                options = combo_options(spec)
                if options is not None and isinstance(value, str) and value not in options:
                    raise HTTPException(422, {"code": "unavailable_backend_option", "node": node["class_type"], "field": field, "value": value})


        available_loras = combo_options(info.get("MiniMaxH3TurboLoRA", {}).get("input", {}).get("required", {}).get("lora_name")) or []
        for lora in compiled["applied_parameters"].get("loras", []):
            if available_loras and lora["name"] not in available_loras:
                raise HTTPException(422, {"code": "unavailable_lora", "name": lora["name"]})

    async def reconcile(row):
        if row["proxy_id"] or row["state"] != "submitting":
            return row
        # A timed-out submit may already be running: recover by unique metadata, never replay POST.
        try:
            response = await upstream("GET", "/api/v2/jobs", params={"limit": 100})
            if response.status_code == 200:
                body = response.json()
                values = body.get("jobs", body.get("items", [])) if isinstance(body, dict) else body
                match = [x for x in values if x.get("metadata") == "h3-api:"+row["id"]]
                if len(match) == 1:
                    store.update(row["id"], "submitted", match[0]["id"])
                    return store.get(row["id"])
        except (HTTPException, ValueError, TypeError):
            pass
        return row

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "h3-workflow-api"}

    @app.get("/ready", dependencies=[Depends(authorized)])
    async def ready():
        policy = await asyncio.to_thread(protected)
        response = await upstream("GET", "/api/v2/health")
        if response.status_code != 200:
            raise HTTPException(503, "Official proxy is unavailable")
        try:
            backend = await http.get(profile["comfy_url"].rstrip("/")+"/queue")
            backend.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, "Protected ComfyUI HTTP API is unavailable") from error
        return {"status": "ready", "protection": policy,
                "resident_cache_enabled": profile.get("resident_cache", False)}

    @app.get("/v1/capabilities", dependencies=[Depends(authorized)])
    async def capabilities():
        return {"profile": profile["name"], "api_version": "0.2.0", "request_schema": GenerateRequest.model_json_schema(),
                "fps": 24, "reference_video": False,
                "supported_modes": ["text_to_video", "reference_images_to_video", "first_last_frame_to_video", "middle_keyframes", "hybrid", "voice_reference", "driving_audio"],
                "audio_modes": ["auto", "native", "reference_only", "lock_source", "remix_source"],
                "runtime": {"resident_cache_enabled": True, "persistent_conditioning_cache": True,
                            "per_job_audio_audit": True, "npu_count": int(profile["npu_count"])},
                "defaults": {"model": profile["model"], "text_model": profile["text_model"], "loras": profile["loras"],
                             "steps": 8, "resolution": "720p", "duration_seconds": 10},
                "protection": {"policy": "owned_cannlab_instance", "thermal_power_protection": False},
                "model_inventory": "/v1/models",
                "limitations": ["Generated voice identity and lip sync are model outputs, not a quality guarantee",
                                "Larger resolutions, longer context and custom sampling increase runtime and memory; parameters are preserved",
                                "lock_source preserves source audio latent and uses final_audio or drive_audio as the soundtrack",
                                "Reference video is outside the original PRO6000 request schema",
                                "External temporary HTTPS access uses polling rather than SSE"]}

    @app.get("/v1/models", dependencies=[Depends(authorized)])
    async def models():
        try:
            response = await http.get(profile["comfy_url"].rstrip("/")+"/object_info")
            response.raise_for_status()
            info = response.json()
        except httpx.HTTPError as error:
            raise HTTPException(503, "ComfyUI node inventory unavailable") from error
        fields = {"models": ("UNETLoader", "unet_name"), "loras": ("MiniMaxH3TurboLoRA", "lora_name"),
                  "encoders": ("CLIPLoader", "clip_name"), "schedulers": ("BasicScheduler", "scheduler")}
        return {key: combo_options(info[cls]["input"]["required"][field]) or [] for key, (cls, field) in fields.items()}

    @app.post("/v1/preview", dependencies=[Depends(authorized)])
    async def preview(body: GenerateRequest):
        try:
            return compile_request(body, profile)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/validate", dependencies=[Depends(authorized)])
    async def validate(body: GenerateRequest):
        compiled = await preview(body)
        policy = await asyncio.to_thread(protected)
        await validate_assets(compiled)
        await check_nodes(compiled)
        return {"valid": True, "generation_submitted": False, "protection": policy, **compiled}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorized)])
    async def submit(body: GenerateRequest, request: Request, idempotency_key: str = Header(min_length=1, max_length=255)):
        request_hash = digest(body.model_dump())
        async with submit_lock:
            try:
                existing = store.lookup(idempotency_key, request_hash)
            except Conflict as error:
                raise HTTPException(409, str(error)) from error
            if existing:
                return public(await reconcile(existing))
            await asyncio.to_thread(protected)
            job_id = str(uuid.uuid4())
            try:
                compiled = compile_request(body, profile, job_id)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            await validate_assets(compiled)
            await check_nodes(compiled)
            # Recheck after network validation, immediately before the only mutating upstream call.
            await asyncio.to_thread(protected)
            try:
                row, claimed = store.claim(idempotency_key, request_hash, compiled, job_id)
            except Conflict as error:
                raise HTTPException(409, str(error)) from error
            if not claimed:
                return public(await reconcile(row))
            try:
                response = await upstream("POST", "/api/v2/jobs", json={"workflow": compiled["workflow"], "metadata": "h3-api:"+job_id},
                                          headers={"Idempotency-Key": "h3-api-"+job_id})
            except HTTPException as error:
                # Keep the durable claim. Even a transport error can follow an accepted generation.
                store.update(job_id, "submitting", error="Submission outcome unknown; poll this job. It will not be submitted again automatically.")
                return public(store.get(job_id))
            if response.status_code not in (200, 201, 202):
                # 5xx/ambiguous server errors may also follow submission. Keep the claim and reconcile.
                unknown = response.status_code >= 500
                store.update(job_id, "submitting" if unknown else "rejected", error=response.text[:2000])
                return JSONResponse(public(store.get(job_id)), status_code=202 if unknown else 422)
            try:
                proxy_id = response.json()["id"]
            except (ValueError, KeyError):
                store.update(job_id, "submitting", error="Proxy response lacks job ID; outcome unknown, do not resubmit")
                return public(store.get(job_id))
            store.update(job_id, "submitted", proxy_id)
            return rewrite(public(store.get(job_id)), request)

    @app.get("/v1/jobs", dependencies=[Depends(authorized)])
    async def list_jobs():
        return {"jobs": [public(x) for x in store.list()]}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorized)])
    async def job(job_id: str, request: Request):
        row = store.get(job_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        row = await reconcile(row)
        result = public(row)
        if row["proxy_id"]:
            try:
                response = await upstream("GET", "/api/v2/jobs/"+quote(row["proxy_id"], safe=""))
                if response.status_code == 200:
                    result["execution"] = rewrite(response.json(), request)
                    # Native SaveVideo is sometimes grouped under ComfyUI's images key.
                    # Preserve the untouched upstream proxy; normalize the outer API output kind.
                    for output in result["execution"].get("outputs", []):
                        if output.get("content_type", "").startswith("video/"):
                            output["type"] = "video"
                    status = result["execution"].get("status", row["state"])
                    result["status"] = status
                    store.update(job_id, status, row["proxy_id"], row["error"])
                else:
                    result["upstream_error"] = response.text[:1000]
            except HTTPException as error:
                result["upstream_error"] = error.detail
        return result

    @app.get("/v1/jobs/{job_id}/workflow", dependencies=[Depends(authorized)])
    async def workflow(job_id: str):
        row = store.get(job_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        return json.loads(row["compiled"])

    @app.get("/v1/jobs/{job_id}/audit", dependencies=[Depends(authorized)])
    async def audio_audit(job_id: str):
        row = store.get(job_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        path = settings.state_dir / "audio-audits" / (row["id"]+".json")
        if not path.exists():
            return JSONResponse({"id": job_id, "audit_status": "pending", "job_status": row["state"]}, status_code=202)
        return {"id": job_id, "audit_status": "complete", **json.loads(path.read_text())}

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorized)])
    async def cancel(job_id: str, request: Request):
        row = store.get(job_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        row = await reconcile(row)
        if not row["proxy_id"]:
            raise HTTPException(409, "Submission outcome unknown: resolve the job identity before cancellation")
        response = await upstream("POST", "/api/v2/jobs/"+quote(row["proxy_id"], safe="")+"/cancel")
        if response.status_code == 404:
            raise HTTPException(501, "Per-job cancellation endpoint/task is unavailable; global interrupt is deliberately not used")
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text[:1500])
        return {"cancellation_requested": True, "message": "Poll the job for terminal state; a request alone does not prove the NPU stopped", "backend": rewrite(response.json(), request)}

    @app.get("/v1/jobs/{job_id}/events", dependencies=[Depends(authorized)])
    async def events(job_id: str):
        row = store.get(job_id)
        if not row or not row["proxy_id"]:
            raise HTTPException(404, "No resolved execution ID")
        url = proxy+"/api/v2/jobs/"+quote(row["proxy_id"], safe="")+"/events"
        return await stream("GET", url, {"Authorization": auth["Authorization"]}, "text/event-stream")

    async def stream(method, url, headers, content_type=None):
        try:
            response = await http.send(http.build_request(method, url, headers=headers, timeout=None), stream=True)
        except httpx.HTTPError as error:
            raise HTTPException(503, "Official proxy unavailable") from error
        if response.status_code >= 400:
            data = await response.aread()
            await response.aclose()
            return Response(data, status_code=response.status_code, media_type=response.headers.get("content-type"))
        async def chunks():
            try:
                async for part in response.aiter_raw():
                    yield part
            finally:
                await response.aclose()
        passthrough = {k: v for k, v in response.headers.items() if k in {"content-length", "content-range", "accept-ranges", "etag", "cache-control"}}
        return StreamingResponse(chunks(), status_code=response.status_code, headers=passthrough,
                                 media_type=content_type or response.headers.get("content-type", "application/octet-stream"))

    @app.post("/v1/assets", dependencies=[Depends(authorized)])
    @app.post("/api/v2/assets", dependencies=[Depends(authorized)], include_in_schema=False)
    async def upload(request: Request, file: UploadFile = File()):
        # No arbitrary URL fetching or model upload in this input-media endpoint.
        kind = file.content_type or "application/octet-stream"
        if not kind.startswith(("image/", "audio/")):
            raise HTTPException(422, "Only image and audio inputs are supported; reference video is not included")
        limit = 100 * 1024 * 1024
        if file.size is not None and file.size > limit:
            raise HTTPException(413, "Input exceeds 100 MiB")
        data = await file.read(limit+1)
        if len(data) > limit:
            raise HTTPException(413, "Input exceeds 100 MiB")
        name = Path(file.filename or "input.bin").name
        response = await upstream("POST", "/api/v2/assets", files={"file": (name, data, kind)},
                                  data={"content_type": kind, "file_path": "input/h3-api/"+uuid.uuid4().hex+"/"+name})
        try:
            body = rewrite(response.json(), request)
        except ValueError:
            raise HTTPException(502, "Invalid official proxy upload response")
        return JSONResponse(body, status_code=response.status_code)

    @app.get("/api/v2/assets/{asset_id}", dependencies=[Depends(authorized)])
    async def asset(asset_id: str, request: Request):
        response = await upstream("GET", "/api/v2/assets/"+quote(asset_id, safe=""))
        return JSONResponse(rewrite(response.json(), request), status_code=response.status_code)

    @app.get("/api/v2/assets/{asset_id}/content", dependencies=[Depends(authorized)])
    async def content(asset_id: str, request: Request):
        headers = auth | ({"Range": request.headers["range"]} if "range" in request.headers else {})
        return await stream("GET", proxy+"/api/v2/assets/"+quote(asset_id, safe="")+"/content", headers)

    # Follow official output links without exposing the raw workflow-submission endpoint.
    @app.get("/api/v2/jobs/{proxy_id}", dependencies=[Depends(authorized)], include_in_schema=False)
    async def official_job_link(proxy_id: str, request: Request):
        row = store.by_proxy(proxy_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        return await job(row["id"], request)

    @app.post("/api/v2/jobs/{proxy_id}/cancel", dependencies=[Depends(authorized)], include_in_schema=False)
    async def official_cancel_link(proxy_id: str, request: Request):
        row = store.by_proxy(proxy_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        return await cancel(row["id"], request)

    @app.get("/api/v2/jobs/{proxy_id}/events", dependencies=[Depends(authorized)], include_in_schema=False)
    async def official_events_link(proxy_id: str):
        row = store.by_proxy(proxy_id)
        if not row:
            raise HTTPException(404, "Unknown job")
        return await events(row["id"])

    return app


def configured_app():
    root = Path(__file__).resolve().parent.parent
    profile = json.loads(Path(os.environ.get("H3_API_PROFILE", root / "config/ascend.json")).read_text())
    state = Path(os.environ.get("H3_API_STATE", root / "runtime"))
    token_file = Path(os.environ.get("H3_API_TOKEN_FILE", state / "token"))
    token = token_file.read_text().strip()
    if not token:
        raise RuntimeError("H3 API token file is empty")
    return make_app(Settings(profile, state, token))
