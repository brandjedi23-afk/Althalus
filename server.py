# server.py
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# -----------------------------
# Paths estables + .env
# -----------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)  # ✅ carga .env al importar server.py

SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR") or (ROOT / "data" / "sessions"))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Token (se evalúa DESPUÉS de load_dotenv)
# -----------------------------
PUBLIC_PATHS = {"/", "/favicon.ico", "/health", "/config", "/docs", "/openapi.json", "/redoc"}

def _get_api_token() -> str:
    # ✅ se lee en runtime (no solo en import) para evitar “token vacío” por orden de carga
    return (os.getenv("DM_API_TOKEN") or "").strip()

def _auth_ok(request: Request, token: str) -> bool:
    # Authorization: Bearer <token>  OR  X-API-KEY: <token>
    auth = request.headers.get("authorization") or ""
    xkey = request.headers.get("x-api-key") or ""

    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() == token
    if xkey:
        return xkey.strip() == token
    return False

# -----------------------------
# Import tolerante del agente
# -----------------------------
AGENT_IMPORT_ERROR: Optional[str] = None
AgentState = None
run_agent_turn = None

try:
    from agent import AgentState as _AgentState, run_agent_turn as _run_agent_turn
    AgentState = _AgentState
    run_agent_turn = _run_agent_turn
except Exception as e:
    AGENT_IMPORT_ERROR = str(e)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DM Agent API", version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PROD: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def require_token(request: Request, call_next):
    path = request.url.path

    # deja pasar endpoints públicos
    if path in PUBLIC_PATHS:
        return await call_next(request)

    token = _get_api_token()
    if not token:
        return JSONResponse(
            status_code=503,
            content={"detail": "Servidor sin DM_API_TOKEN configurado"},
        )

    if not _auth_ok(request, token):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"},
        )

    return await call_next(request)

# -----------------------------
# Modelos
# -----------------------------
class TurnRequest(BaseModel):
    session_id: str
    text: str

class TurnResponse(BaseModel):
    session_id: str
    output: str

class SessionResetRequest(BaseModel):
    session_id: str

# -----------------------------
# Sesiones
# -----------------------------
def _session_path(session_id: str) -> Path:
    safe = "".join(ch for ch in (session_id or "") if ch.isalnum() or ch in ("-", "_")).strip()
    if not safe:
        safe = "default"
    return SESSIONS_DIR / f"{safe}.json"

def load_state(session_id: str):
    # Agent no disponible -> devolvemos None y manejamos arriba
    if not AgentState:
        return None

    p = _session_path(session_id)
    if not p.exists():
        return AgentState()  # type: ignore

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return AgentState()  # type: ignore

    st = AgentState()  # type: ignore
    hist = data.get("history", [])
    if isinstance(hist, list):
        st.history = hist
    return st

def save_state(session_id: str, state) -> None:
    p = _session_path(session_id)
    payload = {"history": getattr(state, "history", [])}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

# -----------------------------
# Runtime checks
# -----------------------------
def _check_runtime_ready() -> Dict[str, Any]:
    if AGENT_IMPORT_ERROR:
        return {"ok": False, "reason": "agent_import_error", "detail": AGENT_IMPORT_ERROR}

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()

    if not api_key:
        return {"ok": False, "reason": "missing_OPENAI_API_KEY", "detail": "Falta OPENAI_API_KEY en entorno/.env"}
    if not model:
        return {"ok": False, "reason": "missing_OPENAI_MODEL", "detail": "Falta OPENAI_MODEL en entorno/.env"}

    return {"ok": True, "model": model}

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "Althalus DM Agent API"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ready": _check_runtime_ready()}

@app.get("/config")
def config() -> Dict[str, Any]:
    # público (no devuelve secretos)
    ready = _check_runtime_ready()
    return {
        "ready": ready,
        "openai_api_key_present": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        "openai_model": (os.getenv("OPENAI_MODEL") or "").strip() or None,
        "dm_api_token_configured": bool(_get_api_token()),
        "sessions_dir": str(SESSIONS_DIR),
        "root": str(ROOT),
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.post("/turn", response_model=TurnResponse)
def turn(req: TurnRequest):
    if AGENT_IMPORT_ERROR or not AgentState or not run_agent_turn:
        raise HTTPException(status_code=503, detail=f"Agente no disponible: {AGENT_IMPORT_ERROR}")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text vacío")

    ready = _check_runtime_ready()
    if not ready.get("ok"):
        raise HTTPException(status_code=503, detail=ready)

    state = load_state(req.session_id)
    if state is None:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    try:
        output = run_agent_turn(req.text, state)
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_MODEL" in msg or "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=500, detail=f"RuntimeError: {msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    save_state(req.session_id, state)
    return TurnResponse(session_id=req.session_id, output=str(output))

# --- RESET: robusto ante bodies vacíos de Actions ---
@app.post("/session/reset")
async def session_reset(
    request: Request,
    session_id: Optional[str] = Query(None),
    req: Optional[SessionResetRequest] = Body(None),
) -> Dict[str, Any]:
    # 1) Preferimos body JSON (si llegó bien)
    sid: Optional[str] = None
    if req and getattr(req, "session_id", None):
        sid = req.session_id

    # 2) Compatibilidad: query param
    if not sid and session_id:
        sid = session_id

    # 3) Tolerancia: intentar leer JSON crudo (Actions a veces manda {})
    if not sid:
        try:
            data = await request.json()
            if isinstance(data, dict):
                sid = data.get("session_id")
        except Exception:
            pass

    if not sid or not str(sid).strip():
        return JSONResponse(
            status_code=422,
            content={"detail": [{"type": "missing", "loc": ["body", "session_id"], "msg": "Field required", "input": {}}]},
        )

    p = _session_path(str(sid))
    if p.exists():
        p.unlink()
    return {"ok": True, "session_id": str(sid)}

@app.post("/session/reset/{session_id}")
def session_reset_path(session_id: str) -> Dict[str, Any]:
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=422, detail="session_id requerido")

    p = _session_path(session_id)
    if p.exists():
        p.unlink()

    return {"ok": True, "session_id": session_id}

@app.get("/session/{session_id}")
def session_dump(session_id: str) -> Dict[str, Any]:
    p = _session_path(session_id)
    if not p.exists():
        return {"ok": True, "session_id": session_id, "history": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"ok": True, "session_id": session_id, "history": data.get("history", [])}
