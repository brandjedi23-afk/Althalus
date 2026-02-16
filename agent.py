import os
import time
import json
import re
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import inspect
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Setup (ROOT -> .env -> client)
# =========================================================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=False)

client = OpenAI()  # lee OPENAI_API_KEY del entorno

def _require_model() -> str:
    m = (os.getenv("OPENAI_MODEL") or "").strip()
    if not m:
        raise RuntimeError("Falta OPENAI_MODEL (ponlo en .env o en variables de entorno).")
    return m

# Rutas ancladas a tu proyecto (evita problemas con uvicorn)
CANON_PATH  = ROOT / "dm" / "canon.json"
LOG_PATH    = ROOT / "data" / "logs" / "session.md"
MEMORY_PATH = ROOT / "data" / "memory.json"

# =========================================================
# Compendios (bestiary + items + spells)
# =========================================================
BESTIARY_PATH = ROOT / "dm" / "bestiary.json"
ITEMS_PATH    = ROOT / "dm" / "items.json"
SPELLS_PATH   = ROOT / "dm" / "spells.json"

# Cache separado para evitar conflicto de formatos:
# - COMPENDIO: {"indexes": {"by_name": ...}, "monsters": {...}}
# - PLANO: {"adult_blue_dragon": {...}, ...}
_BESTIARY_COMP_CACHE: Optional[dict] = None
_BESTIARY_FLAT_CACHE: Optional[dict] = None
_ITEMS_CACHE: Optional[dict] = None
_SPELLS_CACHE: Optional[dict] = None

# =========================================================
# Módulos (PDF) — indexado y consulta (fidelidad de aventura)
# =========================================================
MODULES_DIR   = ROOT / "dm" / "modules"
MODULES_DIR.mkdir(parents=True, exist_ok=True)

# Registro de módulos conocidos (puedes añadir más)
KNOWN_MODULES = {
    "expedition_ruins_greyhawk": {
        "title": "Expedition to the Ruins of Greyhawk",
        "pdf_path": str(ROOT / "dm" / "modules" / "Expedition to the Ruins of Greyhawk.pdf")
    }
}

_MODULE_INDEX_CACHE: Dict[str, dict] = {}

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _module_index_path(module_id: str) -> Path:
    return MODULES_DIR / f"{_slug(module_id)}.index.json"


def _pdf_pages_via_fitz(pdf_path: Path) -> List[str]:
    """
    Extractor robusto para indexado de módulos: PyMuPDF (fitz) únicamente.
    Evita pypdf porque algunos PDFs antiguos rompen el trailer/xref.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages: List[str] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        pages.append(txt)
    doc.close()
    return pages

def _build_module_index(module_id: str, pdf_path: Path, *, chunk_chars: int = 1800, overlap: int = 200) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    try:
        pages = _pdf_pages_via_fitz(pdf_path)  # ✅ principal (robusto)
        if pages and all(not (p or "").strip() for p in pages):
            raise RuntimeError("El PDF parece no contener texto seleccionable (posible escaneado).")
    except Exception as e:
        raise RuntimeError(f"No pude extraer texto del PDF: {e}")

    chunks: List[dict] = []

    for p_idx, txt in enumerate(pages, start=1):
        if not txt:
            continue

        # corta en bloques para retrieval
        start = 0
        cnum = 0
        while start < len(txt):
            end = min(len(txt), start + chunk_chars)
            block = txt[start:end].strip()
            if block:
                chunks.append({
                    "id": f"p{p_idx:03d}_{cnum:03d}",
                    "page": p_idx,
                    "text": block
                })
            cnum += 1
            # overlap para que no se pierdan frases al cortar
            start = end - overlap if end - overlap > start else end

    meta = KNOWN_MODULES.get(module_id, {})
    return {
        "schema_version": "1.0",
        "module_id": module_id,
        "title": meta.get("title", module_id),
        "pdf": str(pdf_path),
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": chunks
    }

def get_module_index(module_id: str) -> Optional[dict]:
    """
    Carga índice desde cache o disco.
    """
    mid = _slug(module_id)
    if mid in _MODULE_INDEX_CACHE:
        return _MODULE_INDEX_CACHE[mid]

    idx_path = _module_index_path(mid)
    if not idx_path.exists():
        return None

    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    _MODULE_INDEX_CACHE[mid] = obj
    return obj

def reset_module_cache() -> None:
    _MODULE_INDEX_CACHE.clear()

def tool_module_load(module_id: str = "expedition_ruins_greyhawk", pdf_path: str = "") -> str:
    """
    Genera (o regenera) el índice del módulo.
    Si pdf_path viene vacío, usa KNOWN_MODULES[module_id].pdf_path
    Guarda en dm/modules/<module_id>.index.json
    """
    mid = _slug(module_id)

    if not pdf_path:
        meta = KNOWN_MODULES.get(mid) or KNOWN_MODULES.get(module_id)
        if not meta:
            return f"Error: module_id '{module_id}' desconocido y pdf_path vacío."
        pdf_path = meta.get("pdf_path", "")

    pdfp = Path(pdf_path)
    if pdfp.exists() and pdfp.parent.as_posix().startswith("/mnt/data"):
        dest = MODULES_DIR / pdfp.name
        if not dest.exists():
            dest.write_bytes(pdfp.read_bytes())
        pdfp = dest

    try:
        idx = _build_module_index(mid, pdfp)
    except Exception as e:
        return f"Error indexando módulo: {e}"

    out_path = _module_index_path(mid)
    out_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    _MODULE_INDEX_CACHE[mid] = idx

    # guarda módulo activo en canon (sin volcar chunks)
    canon = canon_load()
    session = _ensure_dict(canon, "session")

    prev = (session.get("module") or {}).get("progress") or {}
    if not isinstance(prev, dict):
        prev = {}

    progress = {
        "chapter": prev.get("chapter", ""),
        "scene": prev.get("scene", ""),
        "flags": prev.get("flags", []) if isinstance(prev.get("flags", []), list) else [],
    }

    session["module"] = {
        "id": mid,
        "title": idx.get("title", mid),
        "index_path": str(out_path),
        "progress": progress,
    }

    canon_save(canon)

    return json.dumps({
        "ok": True,
        "module_id": mid,
        "title": idx.get("title", mid),
        "index_path": str(out_path),
        "chunks": len(idx.get("chunks", [])),
        "note": "Módulo cargado y marcado como activo en canon.session.module"
    }, ensure_ascii=False, indent=2)

def tool_module_query(query: str, top_k: int = 6, page_min: int = 0, page_max: int = 0) -> str:
    """
    Búsqueda simple por score de términos (substring) sobre chunks.
    Devuelve top_k resultados con id/page/snippet.
    page_min/page_max opcional para acotar (0 = sin límite).
    """
    canon = canon_load()
    mid = _slug(((canon.get("session", {}) or {}).get("module", {}) or {}).get("id", "")) or "expedition_ruins_greyhawk"
    idx = get_module_index(mid)
    if not idx:
        return "Error: no hay índice cargado. Ejecuta module_load() primero."

    q = _norm(query)
    if not q:
        return "Error: query vacío."

    try:
        top_k = int(top_k or 6)
    except Exception:
        top_k = 6
    top_k = max(1, min(top_k, 12))

    terms = [t for t in re.split(r"\s+", q) if t]
    chunks = idx.get("chunks", []) or []

    scored: List[Tuple[int, dict]] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        page = int(ch.get("page", 0) or 0)
        if page_min and page < int(page_min):
            continue
        if page_max and page > int(page_max):
            continue

        text = ch.get("text", "")
        tnorm = _norm(text)

        score = 0
        # scoring súper simple: cuenta ocurrencias por término
        for term in terms:
            if term and term in tnorm:
                score += 2
        # bonus si la query completa aparece
        if q in tnorm:
            score += 3

        if score > 0:
            scored.append((score, ch))

    scored.sort(key=lambda x: (x[0], -int(x[1].get("page", 0) or 0)), reverse=True)
    hits = [ch for _, ch in scored[:top_k]]

    results = []
    for ch in hits:
        txt = ch.get("text", "") or ""
        snippet = txt[:350].replace("\n", " ").strip()
        results.append({
            "id": ch.get("id"),
            "page": ch.get("page"),
            "snippet": snippet
        })

    return json.dumps({
        "module_id": mid,
        "query": query,
        "count": len(results),
        "results": results
    }, ensure_ascii=False, indent=2)

def tool_module_quote(chunk_id: str, max_chars: int = 1200) -> str:
    """
    Devuelve el texto literal de un chunk (para read-aloud o verificación).
    OJO: úsalo con moderación en mesa para no spoilear.
    """
    canon = canon_load()
    mid = _slug(((canon.get("session", {}) or {}).get("module", {}) or {}).get("id", "")) or "expedition_ruins_greyhawk"
    idx = get_module_index(mid)
    if not idx:
        return "Error: no hay índice cargado. Ejecuta module_load() primero."

    cid = (chunk_id or "").strip()
    if not cid:
        return "Error: chunk_id vacío."

    try:
        max_chars = int(max_chars or 1200)
    except Exception:
        max_chars = 1200
    max_chars = max(200, min(max_chars, 4000))

    for ch in (idx.get("chunks", []) or []):
        if isinstance(ch, dict) and ch.get("id") == cid:
            txt = (ch.get("text", "") or "")[:max_chars]
            return json.dumps({
                "module_id": mid,
                "chunk_id": cid,
                "page": ch.get("page", 0),
                "text": txt
            }, ensure_ascii=False, indent=2)

    return f"Error: chunk_id '{cid}' no encontrado."

def tool_module_set_progress(chapter: str = "", scene: str = "", add_flags_json: str = "[]") -> str:
    """
    Actualiza canon.session.module.progress del módulo activo.
    - chapter/scene: si vienen vacíos, se mantienen.
    - add_flags_json: JSON lista de strings para AÑADIR (sin duplicados).
    """
    canon = canon_load()
    session = _ensure_dict(canon, "session")

    mod = session.get("module") or {}
    if not isinstance(mod, dict) or not (mod.get("id") or ""):
        return "Error: no hay módulo activo en canon.session.module. Ejecuta module_load() primero."

    mid = _slug(mod.get("id", ""))
    prev = (mod.get("progress") or {})
    if not isinstance(prev, dict):
        prev = {}

    # parse flags a añadir
    try:
        add_flags = json.loads(add_flags_json or "[]")
        if not isinstance(add_flags, list):
            add_flags = []
    except Exception:
        add_flags = []

    add_flags = [str(f).strip() for f in add_flags if str(f).strip()]
    prev_flags = prev.get("flags", [])
    if not isinstance(prev_flags, list):
        prev_flags = []

    merged_flags = list(dict.fromkeys([*prev_flags, *add_flags]))  # sin duplicados, preserva orden

    progress = {
        "chapter": chapter if str(chapter or "").strip() else prev.get("chapter", ""),
        "scene": scene if str(scene or "").strip() else prev.get("scene", ""),
        "flags": merged_flags,
    }

    # actualiza SOLO progress, preserva el resto del objeto module
    mod["progress"] = progress
    session["module"] = mod

    canon_save(canon)

    st = _get_state()
    if st is not None:
        st.module_progress["module_id"] = mod.get("id") or st.module_progress.get("module_id", "")
        st.module_progress["chapter"] = progress.get("chapter", "")
        st.module_progress["scene"] = progress.get("scene", "")
        st.module_progress["flags"] = progress.get("flags", [])

    return json.dumps({
        "ok": True,
        "module_id": mid,
        "progress": progress
    }, ensure_ascii=False, indent=2)

# =========================================================
# Persistencia: canon / log / memoria
# =========================================================
def canon_load() -> dict:
    if not CANON_PATH.exists():
        return {}
    try:
        return json.loads(CANON_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def canon_save(canon: dict) -> None:
    """
    Guarda dm/canon.json de forma segura:
    - crea directorio si no existe
    - escritura atómica (tmp -> replace) para evitar archivos corruptos
    """
    CANON_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = CANON_PATH.with_suffix(CANON_PATH.suffix + ".tmp")
    data = json.dumps(canon, ensure_ascii=False, indent=2)

    # Escribe a tmp y reemplaza (atómico en la mayoría de OS)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(CANON_PATH)

def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def tool_canon_get(key: str) -> str:
    canon = canon_load()
    # soporte "dot path": party.members[0].name, session.location, etc.
    try:
        return json.dumps(_canon_get_path(canon, key), ensure_ascii=False)
    except Exception:
        return json.dumps(canon.get(key), ensure_ascii=False)

def tool_canon_patch(json_text: str) -> str:
    """
    Aplica un patch JSON (string) al canon con merge recursivo y guardado seguro.
    Soporta que el input venga envuelto en ```json ... ``` (muy típico cuando lo genera el LLM).
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return "Error: json_text vacío."

    raw = json_text.strip()

    # 1) Quitar fences tipo ```json ... ``` o ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()

    # 2) Parsear JSON
    try:
        patch = json.loads(raw)
        if not isinstance(patch, dict):
            return "Error: el patch debe ser un JSON objeto (dict)."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    # 3) Cargar canon actual
    canon = canon_load()
    if not isinstance(canon, dict):
        canon = {}

    # 4) Merge + save (usa tu canon_save atómico)
    canon = deep_merge(canon, patch)
    try:
        canon_save(canon)
    except Exception as e:
        return f"Error: no se pudo guardar canon.json: {e}"

    return "OK. canon.json actualizado (patch aplicado)."

def tool_update_recap(text: str) -> str:
    st = _ACTIVE_STATE
    if st is None:
        return "WARN: no active state"
    st.scene["recap"] = (text or "").strip()
    return "OK: recap actualizado"

def _make_item_instance(item_def: dict, *, qty: int = 1) -> dict:
    """
    Crea una instancia de item para inventario/loot.
    Guardamos lo esencial + un snapshot del texto para no depender del compendio en runtime.
    """
    return {
        "name": item_def.get("name", ""),
        "type": item_def.get("type", ""),
        "rarity": item_def.get("rarity", ""),
        "magic": bool(item_def.get("magic", True)),
        "attunement": bool(item_def.get("attunement", False)),
        "qty": int(qty or 1),
        "text": list(item_def.get("text", []) or []),
        "source": list(item_def.get("source", []) or []),
        "tags": list(item_def.get("tags", []) or []),
    }

def canon_add_item_to_party(canon: dict, member_name: str, item_name: str, qty: int = 1) -> bool:
    member = _find_party_member(canon, member_name)
    if not member:
        return False

    item_def = _get_item_def_by_name(item_name)
    if not item_def:
        return False

    inv = member.setdefault("inventory", [])
    inv.append(_make_item_instance(item_def, qty=qty))
    return True

def canon_add_item_to_loot(canon: dict, item_name: str, qty: int = 1) -> bool:
    item_def = _get_item_def_by_name(item_name)
    if not item_def:
        return False
    loot = canon.setdefault("loot", [])
    loot.append(_make_item_instance(item_def, qty=qty))
    return True

def tool_give_item(member: str, item: str, qty: int = 1) -> str:
    canon = canon_load()
    ok = canon_add_item_to_party(canon, member, item, qty=int(qty or 1))
    if not ok:
        return f"Error: no pude dar '{item}' a '{member}'. (¿existe el PJ y el item en dm/items.json?)"
    canon_save(canon)
    return f"OK. '{item}' x{int(qty or 1)} añadido al inventario de {member}."

def tool_add_loot(item: str, qty: int = 1) -> str:
    canon = canon_load()
    ok = canon_add_item_to_loot(canon, item, qty=int(qty or 1))
    if not ok:
        return f"Error: no pude añadir '{item}' al loot. (¿existe en dm/items.json?)"
    canon_save(canon)
    return f"OK. Loot: '{item}' x{int(qty or 1)} añadido."

def tool_log_event(text: str) -> str:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return "OK. Evento registrado."

def memory_load() -> dict:
    if not MEMORY_PATH.exists():
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text("{}", encoding="utf-8")
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}

def memory_save(mem: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_PATH.with_suffix(MEMORY_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MEMORY_PATH)

def tool_memory_get(key: str) -> str:
    mem = memory_load()
    return json.dumps(mem.get(key), ensure_ascii=False)

def tool_memory_set(key: str, value: str) -> str:
    mem = memory_load()
    mem[key] = value
    memory_save(mem)
    return f"OK. Guardado en memoria['{key}']."

# =========================================================
# Helpers generales
# =========================================================
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _ensure_list(d: dict, key: str) -> list:
    if key not in d or not isinstance(d[key], list):
        d[key] = []
    return d[key]

def _ensure_dict(d: dict, key: str) -> dict:
    if key not in d or not isinstance(d[key], dict):
        d[key] = {}
    return d[key]

def _canon_get_path(obj: Any, path: str) -> Any:
    """
    Soporta:
      - foo.bar
      - foo.bar[0].baz
    """
    if not path or not isinstance(path, str):
        return None
    cur = obj
    parts = [p for p in path.split(".") if p]
    for part in parts:
        m = re.fullmatch(r"([a-zA-Z0-9_\-]+)(?:\[(\d+)\])?", part)
        if not m:
            return None
        key, idx = m.group(1), m.group(2)
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
        if idx is not None:
            if not isinstance(cur, list):
                return None
            i = int(idx)
            if i < 0 or i >= len(cur):
                return None
            cur = cur[i]
    return cur

def _find_party_member(canon: dict, name: str) -> Optional[dict]:
    target = _norm(name)
    for m in canon.get("party", {}).get("members", []):
        if _norm(m.get("name", "")) == target:
            return m
    for m in canon.get("party", {}).get("members", []):
        if target and target in _norm(m.get("name", "")):
            return m
    return None

def _find_enemy(canon: dict, name: str) -> Optional[Tuple[str, dict]]:
    enemies = canon.get("enemies", {}) or {}
    target = _norm(name)
    for k, v in enemies.items():
        if _norm(k) == target:
            return (k, v)
    for k, v in enemies.items():
        if target and target in _norm(k):
            return (k, v)
    return None

def _get_target_container(canon: dict, target: str) -> Optional[Tuple[str, dict, str]]:
    """
    Devuelve (kind, obj, canonical_name)
    kind: "party" | "enemy"
    """
    # 1) Party
    m = _find_party_member(canon, target)
    if m:
        return ("party", m, m.get("name", target))

    # 2) Enemigo ya existente
    fe = _find_enemy(canon, target)
    if fe:
        k, v = fe
        return ("enemy", v, k)

    # 3) Enemigo NO existente -> intenta auto-crear desde bestiary.json
    spawned = _ensure_enemy_from_bestiary(canon, target)
    if spawned:
        k, v = spawned
        return ("enemy", v, k)

    return None

def _unique_enemy_instance_name(canon: dict, base_name: str) -> str:
    enemies = canon.get("enemies", {}) or {}
    if base_name not in enemies:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name} #{i}"
        if candidate not in enemies:
            return candidate
        i += 1


def _speed_to_int(speed_val: Any, default_speed: int = 30) -> int:
    """
    Tu compendio trae speed a veces como dict:
      {"walk": 10, "swim": 40}
    o a veces como int.
    Aquí normalizamos a un entero usable (preferimos walk).
    """
    if isinstance(speed_val, dict):
        if "walk" in speed_val:
            try:
                return int(speed_val.get("walk") or default_speed)
            except Exception:
                return default_speed
        # si no hay walk, usamos el mayor valor numérico
        vals = []
        for v in speed_val.values():
            try:
                vals.append(int(v))
            except Exception:
                pass
        return max(vals) if vals else default_speed

    try:
        return int(speed_val)
    except Exception:
        return default_speed


def _ensure_enemy_from_bestiary(canon: dict, target_name: str) -> Optional[Tuple[str, dict]]:
    """
    Si target_name no existe en canon["enemies"], intenta crearlo desde dm/bestiary.json
    (estructura compendio: indexes.by_name + monsters{}).
    Devuelve (instance_name, enemy_obj) o None.
    """
    # Si ya existe, no hacemos nada
    fe = _find_enemy(canon, target_name)
    if fe:
        return fe

    mon_def = _get_monster_def_by_name(target_name)
    if not mon_def:
        return None

    enemies = _ensure_dict(canon, "enemies")
    base = (mon_def.get("name") or target_name).strip() or target_name
    instance_name = _unique_enemy_instance_name(canon, base)

    enemies[instance_name] = deepcopy(mon_def)
    enemies[instance_name]["_bestiary_name"] = mon_def.get("name", base)

    # -----------------------------
    # Runtime HP unificado: max_hp + hp_current
    # -----------------------------
    hp_raw = enemies[instance_name].get("hp")
    if isinstance(hp_raw, dict):
        max_hp = int(hp_raw.get("avg", 1) or 1)
    else:
        try:
            max_hp = int(hp_raw or 1)
        except Exception:
            max_hp = 1

    enemies[instance_name]["max_hp"] = max_hp
    enemies[instance_name]["hp_current"] = max_hp

    # Condiciones runtime
    if "conditions" not in enemies[instance_name] or not isinstance(enemies[instance_name]["conditions"], list):
        enemies[instance_name]["conditions"] = []

    # Movimiento runtime + speed normalizada (walk / max)
    sp_raw = enemies[instance_name].get("speed", 30)
    enemies[instance_name]["speed"] = _speed_to_int(sp_raw, default_speed=30)
    if "move_left" not in enemies[instance_name]:
        enemies[instance_name]["move_left"] = 0

    return (instance_name, enemies[instance_name])


def _load_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_bestiary() -> Optional[dict]:
    """
    Devuelve el bestiary como COMPENDIO (indexes.by_name + monsters{}).
    """
    global _BESTIARY_COMP_CACHE
    if _BESTIARY_COMP_CACHE is None:
        _BESTIARY_COMP_CACHE = _load_json_file(BESTIARY_PATH)
    return _BESTIARY_COMP_CACHE


def reset_compendium_caches() -> None:
    global _BESTIARY_COMP_CACHE, _BESTIARY_FLAT_CACHE, _ITEMS_CACHE, _SPELLS_CACHE
    _BESTIARY_COMP_CACHE = None
    _BESTIARY_FLAT_CACHE = None
    _ITEMS_CACHE = None
    _SPELLS_CACHE = None


def get_items_compendium() -> Optional[dict]:
    global _ITEMS_CACHE
    if _ITEMS_CACHE is None:
        _ITEMS_CACHE = _load_json_file(ITEMS_PATH)
    return _ITEMS_CACHE


def get_spells_compendium() -> Optional[dict]:
    global _SPELLS_CACHE
    if _SPELLS_CACHE is None:
        _SPELLS_CACHE = _load_json_file(SPELLS_PATH)
    return _SPELLS_CACHE


def _get_spell_def_by_name(name: str) -> Optional[dict]:
    """
    Devuelve el objeto spell (definición) desde dm/spells.json dado un nombre.
    Requiere estructura:
      { "indexes": {"by_name": {...}}, "spells": { "<id>": {...} } }
    """
    comp = get_spells_compendium()
    if not comp:
        return None
    spell_id = _compendium_lookup_by_name(comp, name)
    if not spell_id:
        return None
    return (comp.get("spells") or {}).get(spell_id)


def _compendium_lookup_by_name(comp: dict, name: str) -> Optional[str]:
    """
    Devuelve el ID interno del compendio usando indexes.by_name.
    """
    if not comp or not name:
        return None
    idx = (comp.get("indexes") or {}).get("by_name") or {}
    return idx.get(_norm(name))


def _get_item_def_by_name(name: str) -> Optional[dict]:
    """
    Devuelve el objeto item (definición) desde dm/items.json dado un nombre.
    """
    comp = get_items_compendium()
    if not comp:
        return None
    item_id = _compendium_lookup_by_name(comp, name)
    if not item_id:
        return None
    return (comp.get("items") or {}).get(item_id)


def _get_monster_def_by_name(name: str) -> Optional[dict]:
    """
    (opcional) si quieres mantener simetría con items; si ya tienes esto para bestiary, no lo dupliques.
    """
    comp = get_bestiary()
    if not comp:
        return None
    monster_id = _compendium_lookup_by_name(comp, name)
    if not monster_id:
        return None
    return (comp.get("monsters") or {}).get(monster_id)


def _has_condition(obj: dict, name: str) -> bool:
    name = (name or "").upper()
    for c in obj.get("conditions", []) or []:
        if str(c.get("name", "")).upper() == name:
            return True
    return False


def _ensure_conditions(obj: dict) -> list:
    return _ensure_list(obj, "conditions")

# =========================================================
# XP System (híbrido: progreso + bichos) + escalado por nº PJs
# =========================================================

# Tabla XP por CR (D&D 5e)
CR_XP = {
    0: 10,
    0.125: 25,   # 1/8
    0.25: 50,    # 1/4
    0.5: 100,    # 1/2
    1: 200,
    2: 450,
    3: 700,
    4: 1100,
    5: 1800,
    6: 2300,
    7: 2900,
    8: 3900,
    9: 5000,
    10: 5900,
    11: 7200,
    12: 8400,
    13: 10000,
    14: 11500,
    15: 13000,
    16: 15000,
    17: 18000,
    18: 20000,
    19: 22000,
    20: 25000,
    21: 33000,
    22: 41000,
    23: 50000,
    24: 62000,
    25: 75000,
    26: 90000,
    27: 105000,
    28: 120000,
    29: 135000,
    30: 155000,
}

PROGRESS_XP_PER_PC = {
    "minor": 25,
    "standard": 75,
    "major": 200,
}

def _party_size(canon: dict) -> int:
    members = canon.get("party", {}).get("members", [])
    if isinstance(members, list) and members:
        return len(members)
    # fallback si alguien lo guarda explícito
    try:
        return int(canon.get("party", {}).get("size", 0) or 0)
    except Exception:
        return 0

def _ensure_xp_struct(canon: dict) -> dict:
    xp = canon.get("xp")
    if not isinstance(xp, dict):
        xp = {}
        canon["xp"] = xp
    if "total" not in xp:
        xp["total"] = 0
    if "log" not in xp or not isinstance(xp["log"], list):
        xp["log"] = []
    if "by_character" not in xp or not isinstance(xp["by_character"], dict):
        xp["by_character"] = {}
    return xp

def _add_xp_to_party(canon: dict, amount_total: int, reason: str, meta: Optional[dict] = None) -> dict:
    """
    amount_total: XP total del grupo (antes de repartir) o ya repartido (según como lo uses).
    Aquí lo tratamos como XP DEL GRUPO y opcionalmente lo repartimos a by_character.
    """
    xp = _ensure_xp_struct(canon)
    amount_total = int(amount_total or 0)
    if amount_total <= 0:
        return {"ok": False, "error": "amount_total debe ser > 0"}

    xp["total"] = int(xp.get("total", 0) or 0) + amount_total

    # Reparto por personaje (solo informativo; el total de grupo sigue siendo el “source of truth”)
    members = canon.get("party", {}).get("members", [])
    n = _party_size(canon)
    if n <= 0:
        n = 1
    per_pc = amount_total // n

    if isinstance(members, list):
        for m in members:
            nm = (m.get("name") or "").strip()
            if not nm:
                continue
            xp["by_character"][nm] = int(xp["by_character"].get(nm, 0) or 0) + per_pc

    entry = {
        "amount_total": amount_total,
        "per_pc": per_pc,
        "party_size": n,
        "reason": reason,
    }
    if meta and isinstance(meta, dict):
        entry["meta"] = meta

    xp["log"].append(entry)
    return {"ok": True, **entry, "xp_total_now": xp["total"]}

def _parse_cr(cr_val: Any) -> Optional[float]:
    """
    Acepta:
      - 10
      - "10"
      - "1/2"
      - "1/4"
      - "1/8"
      - 0.5
    """
    if cr_val is None:
        return None
    if isinstance(cr_val, (int, float)):
        return float(cr_val)
    s = str(cr_val).strip()
    if not s:
        return None
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def _xp_for_cr(cr: Any) -> int:
    f = _parse_cr(cr)
    if f is None:
        return 0
    # Normaliza fracciones típicas
    if abs(f - 0.125) < 1e-6:
        f = 0.125
    if abs(f - 0.25) < 1e-6:
        f = 0.25
    if abs(f - 0.5) < 1e-6:
        f = 0.5
    return int(CR_XP.get(f, 0))

def tool_xp_status() -> str:
    canon = canon_load()
    xp = _ensure_xp_struct(canon)
    n = _party_size(canon) or 1
    out = {
        "party_size": n,
        "xp_total": int(xp.get("total", 0) or 0),
        "per_pc_estimate": int(xp.get("total", 0) or 0) // n,
        "by_character": xp.get("by_character", {}),
        "log_tail": (xp.get("log", []) or [])[-10:],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_xp_progress(kind: str, reason: str = "") -> str:
    """
    kind: minor | standard | major
    """
    kind = (kind or "").strip().lower()
    if kind not in PROGRESS_XP_PER_PC:
        return "Error: kind debe ser minor | standard | major"

    canon = canon_load()
    n = _party_size(canon)
    if n <= 0:
        return "Error: no hay party.members en canon; no puedo calcular party_size."

    per_pc = PROGRESS_XP_PER_PC[kind]
    total = per_pc * n

    res = _add_xp_to_party(
        canon,
        amount_total=total,
        reason=f"PROGRESS:{kind} {reason}".strip(),
        meta={"kind": kind, "per_pc": per_pc},
    )
    canon_save(canon)
    return json.dumps(res, ensure_ascii=False, indent=2)

def tool_xp_kill(enemy: str) -> str:
    """
    Otorga XP por un enemigo muerto basándose en su CR.
    Busca en canon.enemies[enemy]; si existe campo cr -> usa tabla.
    """
    canon = canon_load()
    fe = _find_enemy(canon, enemy)
    if not fe:
        return f"Error: enemigo '{enemy}' no existe en canon.enemies"

    name, obj = fe
    cr = obj.get("cr")
    xp_val = _xp_for_cr(cr)
    if xp_val <= 0:
        return f"Error: enemigo '{name}' no tiene CR válido (cr={cr})."

    n = _party_size(canon) or 1
    res = _add_xp_to_party(
        canon,
        amount_total=xp_val,
        reason=f"KILL:{name}",
        meta={"enemy": name, "cr": cr, "xp": xp_val, "per_pc_estimate": xp_val // n},
    )
    canon_save(canon)
    return json.dumps(res, ensure_ascii=False, indent=2)

def tool_level_status() -> str:
    """
    Devuelve nivel estimado del grupo (por XP) usando XP total / nº PJs.
    También devuelve el umbral del siguiente nivel.
    """
    canon = canon_load()
    n = _party_size(canon) or 1
    xp_per_pc = _party_xp_per_pc(canon)
    lvl = _level_for_xp(xp_per_pc)
    nxt = _next_level_threshold(lvl)

    out = {
        "party_size": n,
        "xp_total_group": int(_ensure_xp_struct(canon).get("total", 0) or 0),
        "xp_per_pc": xp_per_pc,
        "level_by_xp": lvl,
        "next_level": (lvl + 1) if lvl < 20 else None,
        "next_level_threshold_xp_per_pc": nxt,
        "xp_needed_per_pc": (max(0, nxt - xp_per_pc) if nxt is not None else 0),
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_level_check_up() -> str:
    """
    Compara el nivel "current_level" guardado en canon (si existe) con el nivel por XP.
    Si hay subida, lo registra y actualiza canon.party.level (opcional).
    """
    canon = canon_load()
    party = canon.get("party", {}) if isinstance(canon.get("party", {}), dict) else {}
    current_level = party.get("level")

    # Si no hay nivel guardado, inferimos el nivel mínimo observado en members (si existe),
    # y si no, usamos el nivel por XP.
    if current_level is None:
        members = party.get("members", [])
        if isinstance(members, list) and members:
            levels = []
            for m in members:
                try:
                    levels.append(int(m.get("level", 0) or 0))
                except Exception:
                    pass
            current_level = max(levels) if levels else None

    xp_level = _level_for_xp(_party_xp_per_pc(canon))
    if current_level is None:
        current_level = xp_level

    try:
        current_level = int(current_level)
    except Exception:
        current_level = xp_level

    up = xp_level > current_level

    # (Opcional) actualizar canon.party.level si sube
    if up:
        party = _ensure_dict(canon, "party")
        party["level"] = xp_level

        # También puedes sincronizar en cada member (solo si quieres)
        members = party.get("members", [])
        if isinstance(members, list):
            for m in members:
                if isinstance(m, dict):
                    m["level"] = xp_level

        # Log dentro del XP log para auditoría
        xp = _ensure_xp_struct(canon)
        xp["log"].append({
            "event": "LEVEL_UP",
            "from": current_level,
            "to": xp_level,
            "xp_per_pc": _party_xp_per_pc(canon),
        })

        canon_save(canon)

    out = {
        "current_level_saved": current_level,
        "level_by_xp": xp_level,
        "level_up": up,
        "new_level": xp_level if up else None
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def tool_level_up_announce() -> str:
    """
    Si hay subida de nivel por XP, actualiza canon.party.level (y members.level) y devuelve
    un mensaje listo para el DM: "Subís a nivel X".
    Si no hay subida, devuelve "Aún no subís de nivel."
    """
    # Reutilizamos la lógica existente
    result = json.loads(tool_level_check_up())

    if result.get("level_up"):
        lvl = result.get("level_by_xp")
        return f"✅ Subís a nivel {lvl}."
    return "Aún no subís de nivel."

# =========================================================
# Level by XP (umbrales por PJ, D&D 5e)
# =========================================================

# Umbrales de XP por PJ para alcanzar ese nivel (PHB 5e).
# Nota: nivel 1 empieza en 0 XP.
LEVEL_XP_THRESHOLDS = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000,
    11: 85000,
    12: 100000,
    13: 120000,
    14: 140000,
    15: 165000,
    16: 195000,
    17: 225000,
    18: 265000,
    19: 305000,
    20: 355000,
}

def _level_for_xp(xp_per_pc: int) -> int:
    """
    Devuelve el nivel (1-20) correspondiente a xp_per_pc según umbrales.
    """
    try:
        x = int(xp_per_pc or 0)
    except Exception:
        x = 0

    lvl = 1
    for L in range(1, 21):
        if x >= LEVEL_XP_THRESHOLDS[L]:
            lvl = L
    return lvl

def _next_level_threshold(level: int) -> Optional[int]:
    if level >= 20:
        return None
    return LEVEL_XP_THRESHOLDS.get(level + 1)

def _party_xp_per_pc(canon: dict) -> int:
    xp = _ensure_xp_struct(canon)
    total = int(xp.get("total", 0) or 0)
    n = _party_size(canon) or 1
    return total // n

# =========================================================
# Spells (Conjuros)
# =========================================================
def tool_spell_info(name: str) -> str:
    """
    Devuelve la definición completa de un spell por nombre (JSON string) desde dm/spells.json.
    Si no hay match exacto en el índice, intenta fallback por substring.
    """
    sp = _get_spell_def_by_name(name)
    if not sp:
        # fallback: intenta encontrar el primer match por substring
        comp = get_spells_compendium() or {}
        spells = (comp.get("spells") or {})
        q = _norm(name or "")
        best = None
        for _id, cand in spells.items():
            if not isinstance(cand, dict):
                continue
            nm = cand.get("name", "")
            if nm and q and q in _norm(nm):
                best = cand
                break
        if not best:
            return f"Error: spell '{name}' no encontrado en dm/spells.json"
        sp = best

    return json.dumps(sp, ensure_ascii=False, indent=2)


def tool_spell_search(query: str = "", limit: int = 10) -> str:
    """
    Busca spells por substring en nombre (fuzzy simple) y devuelve lista de nombres.
    """
    comp = get_spells_compendium() or {}
    spells = (comp.get("spells") or {})

    q = _norm(query or "")
    try:
        limit = int(limit or 10)
    except Exception:
        limit = 10
    limit = max(1, min(limit, 50))

    hits: List[str] = []
    for _id, sp in spells.items():
        if not isinstance(sp, dict):
            continue
        nm = sp.get("name", "")
        if not nm:
            continue
        if not q or q in _norm(nm):
            hits.append(nm)

    hits = sorted(set(hits))[:limit]
    return json.dumps({"query": query, "count": len(hits), "results": hits}, ensure_ascii=False, indent=2)

# =========================================================
# Distancias (bandas)
# =========================================================
def _band_rank(b: str) -> int:
    return {"melee": 0, "short": 1, "medium": 2, "long": 3, "extreme": 4}.get((b or "medium").lower(), 2)

def _get_positions(canon: dict) -> dict:
    return _ensure_dict(canon, "positions")

def tool_set_range(target: str, band: str) -> str:
    band = (band or "").strip().lower()
    if band not in {"melee", "short", "medium", "long", "extreme"}:
        return "band debe ser: melee | short | medium | long | extreme"

    canon = canon_load()
    found = _get_target_container(canon, target)
    cname = found[2] if found else target  # nombre canónico si existe

    pos = _get_positions(canon)
    pos[cname] = band
    canon_save(canon)
    return f"OK. {cname} ahora está en rango '{band}'."

def _canonical_name(canon: dict, name: str) -> str:
    found = _get_target_container(canon, name)
    return found[2] if found else name

def tool_get_range(a: str, b: str) -> str:
    canon = canon_load()
    pos = canon.get("positions", {}) or {}

    ca = _canonical_name(canon, a)
    cb = _canonical_name(canon, b)

    ra = pos.get(ca, "medium")
    rb = pos.get(cb, "medium")

    if ra == "melee" and rb == "melee":
        return "melee"
    rank = max(_band_rank(ra), _band_rank(rb))
    inv = {0: "melee", 1: "short", 2: "medium", 3: "long", 4: "extreme"}
    return inv.get(rank, "medium")

def tool_approach(actor: str) -> str:
    canon = canon_load()
    pos = _get_positions(canon)
    cur = pos.get(actor, "medium")
    pos[actor] = "melee"
    canon_save(canon)
    return f"{actor} se acerca: {cur} → melee"

def tool_retreat(actor: str) -> str:
    canon = canon_load()
    pos = _get_positions(canon)
    cur = pos.get(actor, "melee")
    pos[actor] = "short" if cur == "melee" else "medium"
    canon_save(canon)
    return f"{actor} se aleja: {cur} → {pos[actor]}"


# =========================================================
# Tools: escena
# =========================================================
def tool_start_scene(location: str, hook: str) -> str:
    # 1) Persistencia por sesión (AgentState)
    st = _get_state()
    if st is not None:
        st.scene["location"] = location
        st.scene["current_scene"] = hook
        # no machacamos recap/open_threads si ya existen

    # 2) Compatibilidad hacia atrás (canon.session global)
    canon = canon_load()
    session = _ensure_dict(canon, "session")
    session["location"] = location
    session["current_scene"] = hook
    canon_save(canon)

    return f"Escena iniciada. Lugar: {location}. Gancho: {hook}"


def tool_scene_status() -> str:
    st = _get_state()

    # Preferimos estado por sesión si existe
    if st is not None and isinstance(st.scene, dict):
        out = {
            "location": st.scene.get("location", ""),
            "current_scene": st.scene.get("current_scene", ""),
            "recap": st.scene.get("recap", ""),
            "open_threads": st.scene.get("open_threads", []),
            "flags": st.flags,
            "module_progress": st.module_progress,
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    # Fallback a canon.session (compat)
    canon = canon_load()
    session = canon.get("session", {}) or {}
    out = {
        "location": session.get("location", ""),
        "current_scene": session.get("current_scene", ""),
        "recap": session.get("recap", ""),
        "open_threads": session.get("open_threads", []),
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


# =========================================================
# Tools: dados / checks
# =========================================================
def tool_roll(expr: str) -> str:
    expr = (expr or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", expr)
    if not m:
        return "Formato inválido. Usa: 2d6+3, 1d20-1, d8, etc."
    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    sides = int(sides_str)
    mod = int(mod_str) if mod_str else 0
    if n < 1 or n > 200:
        return "Número de dados fuera de rango (1–200)."
    if sides < 2 or sides > 1000:
        return "Caras fuera de rango (2–1000)."
    rolls = [random.randint(1, sides) for _ in range(n)]
    total = sum(rolls) + mod
    return f"{expr} -> tiradas={rolls}, modificador={mod}, total={total}"

def tool_check(skill: str, dc: int, bonus: int = 0, mode: str = "normal") -> str:
    mode = (mode or "normal").lower()
    if mode not in {"normal", "adv", "dis"}:
        return "Error: mode debe ser 'normal', 'adv' o 'dis'."
    if not isinstance(dc, int) or dc < 1 or dc > 40:
        return "Error: dc fuera de rango (1–40)."
    r1 = random.randint(1, 20)
    r2 = random.randint(1, 20)
    if mode == "normal":
        chosen = r1
        rolls = [r1]
    elif mode == "adv":
        chosen = max(r1, r2)
        rolls = [r1, r2]
    else:
        chosen = min(r1, r2)
        rolls = [r1, r2]
    total = chosen + int(bonus)
    success = total >= dc
    return (
        f"CHECK {skill} | DC {dc} | bonus {bonus} | mode {mode}\n"
        f"tiradas={rolls} -> elegido={chosen} -> total={total}\n"
        f"resultado={'ÉXITO' if success else 'FALLO'}"
    )

# ----------------------------
# Skill check automático (abilities + prof_bonus + skill_proficiencies)
# ----------------------------
SKILL_TO_ABILITY = {
    # STR
    "athletics": "str",
    # DEX
    "acrobatics": "dex",
    "sleight of hand": "dex",
    "sleight_of_hand": "dex",
    "stealth": "dex",
    # INT
    "arcana": "int",
    "history": "int",
    "investigation": "int",
    "nature": "int",
    "religion": "int",
    # WIS
    "animal handling": "wis",
    "animal_handling": "wis",
    "insight": "wis",
    "medicine": "wis",
    "perception": "wis",
    "survival": "wis",
    # CHA
    "deception": "cha",
    "intimidation": "cha",
    "performance": "cha",
    "persuasion": "cha",
}

def _ability_mod(score: int) -> int:
    try:
        s = int(score)
    except Exception:
        s = 10
    return (s - 10) // 2

def tool_skill_check(actor: str, skill: str, dc: int, mode: str = "normal", extra_bonus: int = 0) -> str:
    """
    Skill check automático para PJ:
      bonus = mod(ability asociada) + (prof_bonus si competente) + extra_bonus
    Devuelve SIEMPRE desglose + la tirada (vía tool_check).
    """
    canon = canon_load()
    m = _find_party_member(canon, actor)
    if not m:
        return f"Error: '{actor}' no es un PJ (party member)."

    # Normaliza skill
    sk_raw = (skill or "").strip().lower().replace("-", " ")
    sk_raw = " ".join(sk_raw.split())
    sk_key = sk_raw.replace(" ", "_")
    sk_space = sk_raw

    ability_key = SKILL_TO_ABILITY.get(sk_raw) or SKILL_TO_ABILITY.get(sk_key) or SKILL_TO_ABILITY.get(sk_space)
    if not ability_key:
        return (
            f"Error: habilidad '{skill}' no reconocida.\n"
            f"Usa check(skill, dc, bonus, mode) o añade el mapeo en SKILL_TO_ABILITY."
        )

    abilities = m.get("abilities", {}) if isinstance(m.get("abilities", {}), dict) else {}
    score = abilities.get(ability_key, 10)
    mod = _ability_mod(score)

    pb = int(m.get("prof_bonus", 0) or 0)

    profs = m.get("skill_proficiencies", [])
    if not isinstance(profs, list):
        profs = []
    profs_norm = set(" ".join(str(x).strip().lower().replace("-", " ").split()) for x in profs)

    proficient = (sk_raw in profs_norm) or (sk_space in profs_norm) or (sk_key in profs_norm)
    prof_part = pb if proficient else 0
    bonus = mod + prof_part + int(extra_bonus)

    breakdown = (
        f"SKILL_CHECK {m.get('name')} -> {skill} ({ability_key.upper()})\n"
        f"score {ability_key}={score} mod={mod} | prof_bonus={pb} | proficient={proficient} (+{prof_part}) | extra={int(extra_bonus)}\n"
        f"TOTAL BONUS = {bonus}\n"
    )

    # Tirada con tu tool_check para que incluya las tiradas
    label = f"{skill} ({ability_key.upper()}){' [PROF]' if proficient else ''}"
    roll_out = tool_check(skill=label, dc=int(dc), bonus=int(bonus), mode=mode)

    return breakdown + roll_out

# =========================================================
# Tools: condiciones / HP
# =========================================================
def tool_apply_condition(target: str, condition: str, rounds: int = 1) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    conds = _ensure_conditions(obj)

    name = (condition or "").strip().upper()
    rounds = int(rounds)
    if not name:
        return "condition no puede estar vacío."
    if rounds < 1 or rounds > 99:
        return "rounds fuera de rango (1–99)."

    for c in conds:
        if str(c.get("name", "")).upper() == name:
            c["remaining"] = max(int(c.get("remaining", 0)), rounds)
            canon_save(canon)
            return f"OK. {cname} mantiene {name} ({c['remaining']} rondas)."

    conds.append({"name": name, "remaining": rounds})
    canon_save(canon)
    return f"OK. {cname} gana {name} ({rounds} rondas)."

def tool_remove_condition(target: str, condition: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    name = (condition or "").strip().upper()
    if not name:
        return "condition no puede estar vacío."
    before = len(obj.get("conditions", []) or [])
    obj["conditions"] = [c for c in (obj.get("conditions", []) or []) if str(c.get("name", "")).upper() != name]
    canon_save(canon)
    removed = before - len(obj["conditions"])
    return f"OK. Quitado {name} de {cname}." if removed else f"{cname} no tenía {name}."

def tool_target_status(target: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    kind, obj, cname = found
    conds = obj.get("conditions", []) or []

    if kind == "party":
        hp = obj.get("hp", 0)
        max_hp = obj.get("max_hp", hp)
        ac = obj.get("ac", 10)
        stable = obj.get("stable", True)
        header = f"{cname} (PJ) HP {hp}/{max_hp} CA {ac} estable={stable}"
    else:
        if "hp_current" in obj and "max_hp" in obj:
            header = f"{cname} (ENEMIGO) HP {obj.get('hp_current','?')}/{obj.get('max_hp','?')} CA {obj.get('ac','?')}"
        else:
            header = f"{cname} (ENEMIGO) HP {obj.get('hp','?')} CA {obj.get('ac','?')}"

    if not conds:
        return header + "\nCondiciones: (ninguna)"
    pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
    return header + f"\nCondiciones: {pretty}"

def tool_conditions_status() -> str:
    canon = canon_load()
    lines = []
    for m in canon.get("party", {}).get("members", []):
        conds = m.get("conditions", []) or []
        if conds:
            pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
            lines.append(f"PJ {m.get('name')}: {pretty}")
    for name, e in (canon.get("enemies", {}) or {}).items():
        conds = e.get("conditions", []) or []
        if conds:
            pretty = ", ".join([f"{c['name']}({c.get('remaining','?')})" for c in conds])
            lines.append(f"EN {name}: {pretty}")
    return "\n".join(lines) if lines else "No hay condiciones activas."

def tool_heal(target: str, amount: int) -> str:
    canon = canon_load()
    m = _find_party_member(canon, target)
    if not m:
        return f"Objetivo '{target}' no es un PJ."
    amount = int(amount)
    if amount <= 0:
        return "amount debe ser > 0"
    hp = int(m.get("hp", 0))
    max_hp = int(m.get("max_hp", hp))
    new_hp = min(max_hp, hp + amount)
    m["hp"] = new_hp
    if new_hp > 0:
        m["stable"] = True
    canon_save(canon)
    return f"Curación: {m.get('name')} {hp} → {new_hp} (max {max_hp})"

def tool_damage(target: str, amount: int) -> str:
    canon = canon_load()
    m = _find_party_member(canon, target)
    if not m:
        return f"Objetivo '{target}' no es un PJ."
    amount = int(amount)
    if amount <= 0:
        return "amount debe ser > 0"
    hp = int(m.get("hp", 0))
    new_hp = max(0, hp - amount)
    m["hp"] = new_hp
    if new_hp == 0:
        m["stable"] = False
    canon_save(canon)
    status = "INCONSCIENTE (inestable)" if new_hp == 0 else "OK"
    return f"Daño: {m.get('name')} {hp} → {new_hp} ({status})"

def tool_stabilize(target: str) -> str:
    canon = canon_load()
    m = _find_party_member(canon, target)
    if not m:
        return f"Objetivo '{target}' no es un PJ."
    hp = int(m.get("hp", 0))
    if hp > 0:
        return f"{m.get('name')} no está a 0 HP; no necesita estabilización."
    m["stable"] = True
    canon_save(canon)
    return f"{m.get('name')} queda ESTABLE a 0 HP (inconsciente)."

def tool_rest(kind: str = "short") -> str:
    kind = (kind or "short").strip().lower()
    if kind not in {"short", "long"}:
        return "kind debe ser 'short' o 'long'"
    canon = canon_load()
    members = canon.get("party", {}).get("members", [])
    if not members:
        return "No hay miembros del grupo."
    lines = [f"Descanso: {kind}"]
    for m in members:
        name = m.get("name", "?")
        hp = int(m.get("hp", 0))
        max_hp = int(m.get("max_hp", hp))
        if kind == "long":
            m["hp"] = max_hp
            m["stable"] = True
            lines.append(f"- {name}: {hp} → {max_hp}")
        else:
            heal_amt = max(1, max_hp // 4)
            new_hp = min(max_hp, hp + heal_amt)
            m["hp"] = new_hp
            if new_hp > 0:
                m["stable"] = True
            lines.append(f"- {name}: {hp} → {new_hp} (+{heal_amt})")
    canon_save(canon)
    return "\n".join(lines)


# =========================================================
# Tools: movimiento
# =========================================================
def _get_speed(obj: dict, default_speed: int = 30) -> int:
    try:
        return int(obj.get("speed", default_speed))
    except Exception:
        return default_speed

def _set_move_left(obj: dict, value: int) -> None:
    obj["move_left"] = max(0, int(value))

def _get_move_left(obj: dict) -> int:
    try:
        return int(obj.get("move_left", 0))
    except Exception:
        return 0

def tool_start_turn() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    idx = int(combat.get("turn_index", 0))
    order = combat.get("order", []) or []
    if not order or idx >= len(order):
        return "Orden de combate inválido."
    actor_name = order[idx]["name"]
    found = _get_target_container(canon, actor_name)
    if not found:
        return f"No encuentro al actor actual '{actor_name}'."
    _, obj, cname = found
    speed = _get_speed(obj, 30)
    _set_move_left(obj, speed)
    canon_save(canon)
    return f"Inicio de turno: {cname}. Movimiento disponible: {speed} ft."

def tool_move(target: str, feet: int) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    feet = int(feet)
    if feet <= 0:
        return "feet debe ser > 0"
    if _has_condition(obj, "RESTRAINED"):
        return f"{cname} está RESTRAINED y no puede moverse."
    move_left = _get_move_left(obj)
    if move_left <= 0:
        return f"{cname} no tiene movimiento disponible (move_left=0)."
    if feet > move_left:
        return f"{cname} solo tiene {move_left} ft de movimiento; no puede mover {feet}."
    _set_move_left(obj, move_left - feet)
    canon_save(canon)
    return f"{cname} se mueve {feet} ft. Movimiento restante: {_get_move_left(obj)} ft."

def tool_stand_up(target: str) -> str:
    canon = canon_load()
    found = _get_target_container(canon, target)
    if not found:
        return f"No encuentro el objetivo '{target}'."
    _, obj, cname = found
    if not _has_condition(obj, "PRONE"):
        return f"{cname} no está PRONE."
    if _has_condition(obj, "RESTRAINED"):
        return f"{cname} está RESTRAINED y no puede levantarse."
    speed = _get_speed(obj, 30)
    cost = max(5, speed // 2)
    move_left = _get_move_left(obj)
    if move_left < cost:
        return f"{cname} necesita {cost} ft para levantarse, pero solo tiene {move_left}."
    _set_move_left(obj, move_left - cost)
    obj["conditions"] = [c for c in (obj.get("conditions", []) or []) if str(c.get("name","")).upper() != "PRONE"]
    canon_save(canon)
    return f"{cname} se levanta (gasta {cost} ft). Movimiento restante: {_get_move_left(obj)} ft."


# =========================================================
# Tools: combate (iniciativa + turnos + tick condiciones)
# =========================================================
def tool_start_combat(combatants_json: str) -> str:
    try:
        combatants = json.loads(combatants_json)
        if not isinstance(combatants, list) or not combatants:
            return "Error: combatants_json debe ser una lista JSON no vacía."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    order = []
    for c in combatants:
        if not isinstance(c, dict) or "name" not in c:
            return "Error: cada combatiente debe ser objeto con al menos 'name'."
        name = str(c.get("name"))
        side = str(c.get("side", "unknown"))
        init_bonus = int(c.get("init_bonus", 0))
        roll = random.randint(1, 20)
        init_total = roll + init_bonus
        order.append({
            "name": name,
            "side": side,
            "init_bonus": init_bonus,
            "init_roll": roll,
            "init_total": init_total
        })

    order.sort(key=lambda x: (x["init_total"], x["init_roll"], x["name"]), reverse=True)

    canon = canon_load()
    canon["combat"] = {
        "active": True,
        "round": 1,
        "turn_index": 0,
        "order": order,
        "bonus_attack_available": False,
        "bonus_attack_owner": None,
        "pending_reaction": None,
    }
    canon_save(canon)

    current = order[0]
    pretty = "\n".join([f"{i+1}. {o['name']} ({o['side']}) init={o['init_total']} [{o['init_roll']}+{o['init_bonus']}]" for i, o in enumerate(order)])
    return (
        "Combate iniciado.\n"
        f"Ronda: 1\n"
        f"Turno actual: {current['name']} ({current['side']})\n"
        "Orden de iniciativa:\n"
        f"{pretty}"
    )

def tool_start_combat_quick(enemies_json: str) -> str:
    """
    Inicia combate usando los PJs del canon + una lista de enemigos en JSON.

    enemies_json ejemplo:
    [
      {"name":"Acolyte", "init_bonus":2},
      {"name":"Aboleth", "init_bonus":0}
    ]

    Si un enemigo no existe en canon["enemies"], intentamos auto-crearlo desde dm/bestiary.json
    (vía _ensure_enemy_from_bestiary). Si tampoco existe en bestiary, metemos defaults.
    """
    try:
        enemies = json.loads(enemies_json)
        if not isinstance(enemies, list) or not enemies:
            return "Error: enemies_json debe ser una lista JSON no vacía."
    except Exception as e:
        return f"Error: JSON inválido: {e}"

    canon = canon_load()

    # Asegurar que el contenedor de enemigos es un dict
    enemies_db = canon.get("enemies", {})
    if not isinstance(enemies_db, dict):
        enemies_db = {}
        canon["enemies"] = enemies_db

    # Asegurar que los enemigos existen en canon["enemies"]
    for e in enemies:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue

        # 1) Si ya existe, ok
        if name in enemies_db:
            continue

        # 2) Si no existe, intenta auto-crear desde bestiary (por nombre)
        spawned = _ensure_enemy_from_bestiary(canon, name)
        if spawned:
            inst_name, _obj = spawned
            # Si el bestiary te crea con nombre distinto (por sufijo #2, etc),
            # y el usuario te pidió un "name" concreto, reflejamos el nombre en la lista
            # para que iniciativa use el nombre real en canon.
            if inst_name != name:
                e["name"] = inst_name
            continue

        # 3) Fallback defaults (si no está en bestiary)
        enemies_db[name] = {"speed": 30, "ac": 13, "hp": 11, "hp_current": 11, "conditions": []}

    canon["enemies"] = enemies_db
    canon_save(canon)

    members = canon.get("party", {}).get("members", [])
    if not members:
        return "Error: no hay miembros del grupo en el canon."

    combatants = []
    for m in members:
        combatants.append({
            "name": m.get("name"),
            "side": "party",
            "init_bonus": int(m.get("init_bonus", 0)),
        })

    for e in enemies:
        if not isinstance(e, dict):
            continue
        combatants.append({
            "name": e.get("name"),
            "side": "enemy",
            "init_bonus": int(e.get("init_bonus", 0)),
        })

    return tool_start_combat(json.dumps(combatants, ensure_ascii=False))


def tool_combat_status() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    order = combat.get("order", []) or []
    idx = int(combat.get("turn_index", 0))
    rnd = int(combat.get("round", 1))
    current = order[idx] if order else {"name": "?", "side": "?"}
    pretty = "\n".join([f"{i+1}. {o['name']} ({o['side']}) init={o['init_total']}" for i, o in enumerate(order)])
    return f"Ronda: {rnd}\nTurno actual: {current['name']} ({current['side']})\nOrden:\n{pretty}"

def _tick_conditions_for(canon: dict, name: str) -> None:
    found = _get_target_container(canon, name)
    if not found:
        return
    _, obj, _ = found
    conds = _ensure_conditions(obj)

    new_conds = []
    for c in conds:
        rem = int(c.get("remaining", 0)) - 1
        if rem > 0:
            new_conds.append({"name": c.get("name", ""), "remaining": rem})
    obj["conditions"] = new_conds


def tool_next_turn() -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    order = combat.get("order", []) or []
    if not order:
        return "Combate activo pero sin orden de iniciativa."

    idx = int(combat.get("turn_index", 0))
    rnd = int(combat.get("round", 1))

    current_actor = order[idx]["name"]
    _tick_conditions_for(canon, current_actor)  # ✅ ya no guarda aparte

    idx += 1
    if idx >= len(order):
        idx = 0
        rnd += 1

    combat["turn_index"] = idx
    combat["round"] = rnd
    canon["combat"] = combat

    # ✅ un solo guardado al final
    canon_save(canon)

    start_msg = tool_start_turn()
    current = order[idx]
    return f"{start_msg}\nRonda: {rnd}\nTurno: {current['name']} ({current['side']})"

def tool_end_combat() -> str:
    canon = canon_load()
    canon["combat"] = {"active": False}
    canon_save(canon)
    return "Combate finalizado."


# =========================================================
# Tools: ataque (simple, pero estable) + bonus attack flag
# =========================================================
def _avg_damage(expr: str) -> float:
    expr = (expr or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", expr)
    if not m:
        return 0.0
    n_str, sides_str, mod_str = m.groups()
    n = int(n_str) if n_str else 1
    sides = int(sides_str)
    mod = int(mod_str) if mod_str else 0
    return n * (1 + sides) / 2.0 + mod

def _hit_prob(target_ac: int, atk_bonus: int, mode: str) -> float:
    mode = (mode or "normal").lower()
    roll_needed = target_ac - atk_bonus
    if roll_needed <= 1:
        p = 1.0
    elif roll_needed > 20:
        p = 0.0
    else:
        p = (21 - roll_needed) / 20.0
    if mode == "adv":
        return 1.0 - (1.0 - p) ** 2
    if mode == "dis":
        return p ** 2
    return p

def _cover_bonus(cover: str) -> int:
    return {"none": 0, "half": 2, "three-quarters": 5, "total": 999}.get((cover or "none").lower(), 0)

def tool_attack(
    attacker: str,
    target: str,
    bonus: int,
    damage: str = "1d8",
    attack_type: str = "melee",
    power_shot: bool = False,
    power_attack: bool = False,
    auto_power: bool = True,
    crit_range: int = 20
) -> str:
    attack_type = (attack_type or "melee").lower()
    if attack_type not in {"melee", "ranged"}:
        return "attack_type debe ser 'melee' o 'ranged'"

    canon = canon_load()
    found_a = _get_target_container(canon, attacker)
    found_t = _get_target_container(canon, target)
    if not found_a:
        return f"No encuentro atacante '{attacker}'."
    if not found_t or found_t[0] != "enemy":
        return f"Objetivo '{target}' no existe como enemigo en canon['enemies']."

    _, attacker_obj, attacker_name = found_a
    _, target_obj, target_name = found_t

    cover = (canon.get("cover", {}) or {}).get(target_name, "none")
    cb = _cover_bonus(cover)
    if cb >= 999:
        return f"{target_name} tiene cobertura TOTAL. No puede ser atacado."

    base_ac = int(target_obj.get("ac", 10))
    target_ac = base_ac + cb

    mode = "normal"
    if _has_condition(target_obj, "PRONE"):
        mode = "adv" if attack_type == "melee" else "dis"

    if attack_type == "melee":
        rng = tool_get_range(attacker_name, target_name)
        if rng != "melee":
            return f"No estás en melee con {target_name}. Usa approach() o set_range()."

    dmg_bonus = 0
    atk_bonus = int(bonus)

    if power_shot and attack_type == "ranged":
        atk_bonus -= 5
        dmg_bonus += 10

    use_power_attack = False
    if attack_type == "melee":
        if power_attack:
            use_power_attack = True
        elif auto_power:
            base_avg = _avg_damage(damage)
            p_base = _hit_prob(target_ac, atk_bonus, mode)
            p_pow = _hit_prob(target_ac, atk_bonus - 5, mode)
            ev_base = p_base * base_avg
            ev_pow = p_pow * (base_avg + 10)
            use_power_attack = ev_pow > ev_base

    if use_power_attack:
        atk_bonus -= 5
        dmg_bonus += 10

    roll = random.randint(1, 20)
    total = roll + atk_bonus

    # Reglas base d20:
    # - 1 natural = fallo automático
    # - 20 natural = acierto automático y crítico (en este motor)
    if roll == 1:
        hit = False
        crit = False
    else:
        hit = (roll == 20) or (total >= target_ac)
        crit = hit and ((roll == 20) or (roll >= int(crit_range)))

    log = [
        f"{attacker_name} ataca a {target_name} ({attack_type})",
        f"Tirada: d20={roll} + {atk_bonus} = {total} vs CA {target_ac} (base {base_ac} + cover {cover})"
    ]

    if not hit:
        canon_save(canon)
        return "\n".join(log + ["Resultado: FALLO"])

    base = tool_roll(damage)
    try:
        dmg = int(base.split("total=")[-1])
    except Exception:
        dmg = 0

    if crit:
        extra = tool_roll(damage)
        try:
            dmg += int(extra.split("total=")[-1])
        except Exception:
            pass

    dmg += dmg_bonus

    # hp runtime
    if "hp_current" in target_obj:
        target_hp = int(target_obj.get("hp_current", 0) or 0)
        new_hp = max(0, target_hp - dmg)
        target_obj["hp_current"] = new_hp
    else:
        try:
            target_hp = int(target_obj.get("hp", 0) or 0)
        except Exception:
            target_hp = 0
        new_hp = max(0, target_hp - dmg)
        target_obj["hp"] = new_hp

    killed = (new_hp <= 0)

    if killed:
        # XP por kill automático
        cr = target_obj.get("cr")
        xp_val = _xp_for_cr(cr)
        if xp_val > 0:
            _add_xp_to_party(canon, xp_val, reason=f"KILL:{target_name}", meta={"cr": cr})
            log.append(f"XP: +{xp_val} (CR {cr}) al grupo")

    if power_shot and attack_type == "ranged":
        log.append("Power Shot (SS-style): -5 al ataque, +10 al daño")
    if use_power_attack and attack_type == "melee":
        log.append("Power Attack (GWM-style): -5 al ataque, +10 al daño")

    log.append(f"Impacto{' CRÍTICO' if crit else ''}: daño {dmg}")
    log.append(f"HP de {target_name}: {target_hp} → {new_hp}")

    combat = canon.get("combat", {}) or {}
    if combat.get("active") and attack_type == "melee" and (crit or killed):
        combat["bonus_attack_available"] = True
        combat["bonus_attack_owner"] = attacker_name
        canon["combat"] = combat
        log.append("Ataque extra (Bonus Action) DISPONIBLE por crítico/muerte")

    if killed and (canon.get("combat", {}) or {}).get("active"):
        order = canon["combat"].get("order", []) or []
        canon["combat"]["order"] = [o for o in order if o.get("name") != target_name]
        log.append(f"{target_name} cae derrotado.")

    canon_save(canon)
    return "\n".join(log)

def tool_bonus_attack(attacker: str, target: str, bonus: int, damage: str = "1d8") -> str:
    canon = canon_load()
    combat = canon.get("combat", {}) or {}
    if not combat.get("active"):
        return "No hay combate activo."
    if not combat.get("bonus_attack_available"):
        return "No hay ataque extra disponible."
    owner = combat.get("bonus_attack_owner")
    if _norm(owner) != _norm(attacker):
        return f"El ataque extra disponible es de {owner}, no de {attacker}."

    combat["bonus_attack_available"] = False
    combat["bonus_attack_owner"] = None
    canon["combat"] = combat
    canon_save(canon)

    result = tool_attack(
        attacker=attacker,
        target=target,
        bonus=bonus,
        damage=damage,
        attack_type="melee",
        power_attack=False,
        auto_power=True,
    )
    return "BONUS ATTACK\n" + result


# =========================================================
# Tools registry
# =========================================================
TOOLS: Dict[str, Callable[..., str]] = {
    "canon_get": tool_canon_get,
    "canon_patch": tool_canon_patch,
    "memory_get": tool_memory_get,
    "memory_set": tool_memory_set,
    "log_event": tool_log_event,
    "give_item": tool_give_item,
    "add_loot": tool_add_loot,

    "spell_info": tool_spell_info,
    "spell_search": tool_spell_search,

    "module_load": tool_module_load,
    "module_query": tool_module_query,
    "module_quote": tool_module_quote,
    "module_set_progress": tool_module_set_progress,
    "update_recap": tool_update_recap,

    "start_scene": tool_start_scene,
    "scene_status": tool_scene_status,

    "roll": tool_roll,
    "check": tool_check,
    "skill_check": tool_skill_check,

    "start_turn": tool_start_turn,
    "move": tool_move,
    "stand_up": tool_stand_up,
    "set_range": tool_set_range,
    "get_range": tool_get_range,
    "approach": tool_approach,
    "retreat": tool_retreat,

    "apply_condition": tool_apply_condition,
    "remove_condition": tool_remove_condition,
    "target_status": tool_target_status,
    "conditions_status": tool_conditions_status,
    "heal": tool_heal,
    "damage": tool_damage,
    "stabilize": tool_stabilize,
    "rest": tool_rest,

    "start_combat": tool_start_combat,
    "start_combat_quick": tool_start_combat_quick,
    "combat_status": tool_combat_status,
    "next_turn": tool_next_turn,
    "end_combat": tool_end_combat,

    "attack": tool_attack,
    "bonus_attack": tool_bonus_attack,

    "xp_status": tool_xp_status,
    "xp_progress": tool_xp_progress,
    "xp_kill": tool_xp_kill,
    "level_status": tool_level_status,
    "level_check_up": tool_level_check_up,
    "level_up_announce": tool_level_up_announce,
}

# =========================================================
# Tool schemas
# =========================================================
def _schema(name: str, desc: str, props: dict, required: list) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }

TOOL_SCHEMAS = [
    _schema("canon_get", "Lee una clave del canon (estado) y devuelve JSON string.", {"key": {"type": "string"}}, ["key"]),
    _schema("canon_patch", "Aplica un patch JSON (string) al canon (merge recursivo).", {"json_text": {"type": "string"}}, ["json_text"]),
    _schema("log_event", "Añade un evento breve al log de sesión.", {"text": {"type": "string"}}, ["text"]),
    
    _schema("give_item", "Da un objeto mágico/ítem a un miembro del grupo usando dm/items.json.", {
        "member": {"type": "string"},
        "item": {"type": "string"},
        "qty": {"type": "integer"}
    }, ["member", "item"]),

    _schema("add_loot", "Añade un objeto mágico/ítem al loot común usando dm/items.json.", {
        "item": {"type": "string"},
        "qty": {"type": "integer"}
    }, ["item"]),

    _schema("spell_info", "Devuelve la definición completa de un spell desde dm/spells.json.", {
        "name": {"type": "string"}
    }, ["name"]),

    _schema("spell_search", "Busca spells por nombre (substring) en dm/spells.json.", {
        "query": {"type": "string"},
        "limit": {"type": "integer"}
    }, []),

    _schema("module_load", "Indexa y carga un módulo PDF y lo marca activo en canon.session.module.", {
        "module_id": {"type": "string"},
        "pdf_path": {"type": "string"}
    }, []),

    _schema("module_query", "Busca texto del módulo activo (top_k resultados con id/página/snippet).", {
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
        "page_min": {"type": "integer"},
        "page_max": {"type": "integer"}
    }, ["query"]),

    _schema("module_quote", "Devuelve el texto literal de un chunk por id (para read-aloud/verificación).", {
        "chunk_id": {"type": "string"},
        "max_chars": {"type": "integer"}
    }, ["chunk_id"]),

    _schema("module_set_progress", "Actualiza progreso del módulo en canon.session.module.progress.", {
        "chapter": {"type": "string"},
        "scene": {"type": "string"},
        "add_flags_json": {"type": "string"}
    }, []),

    _schema("update_recap", "Actualiza el resumen persistente de la escena (scene.recap) para continuidad sin releer todo el history.",
        {"text": {"type": "string"}},
        ["text"]
    ),

    _schema("start_scene", "Inicializa escena en canon.session.", {"location": {"type": "string"}, "hook": {"type": "string"}}, ["location", "hook"]),
    _schema("scene_status", "Devuelve estado de escena (JSON).", {}, []),

    _schema("roll", "Tira dados: 2d6+3, 1d20+5, d8, etc.", {"expr": {"type": "string"}}, ["expr"]),
    _schema("check", "Skill check vs DC (normal/adv/dis).", {
        "skill": {"type": "string"},
        "dc": {"type": "integer"},
        "bonus": {"type": "integer"},
        "mode": {"type": "string"}
    }, ["skill", "dc"]),
    _schema("skill_check", "Tirada de habilidad para un PJ calculando bonus automático (mod + PB si competente).", {
        "actor": {"type": "string"},
        "skill": {"type": "string"},
        "dc": {"type": "integer"},
        "mode": {"type": "string"},
        "extra_bonus": {"type": "integer"}
    }, ["actor", "skill", "dc"]),

    _schema("start_combat", "Inicia combate con lista JSON de combatientes.", {"combatants_json": {"type": "string"}}, ["combatants_json"]),
    _schema("start_combat_quick", "Inicia combate usando PJs + enemigos JSON.", {"enemies_json": {"type": "string"}}, ["enemies_json"]),
    _schema("combat_status", "Estado del combate.", {}, []),
    _schema("next_turn", "Avanza turno y hace tick de condiciones.", {}, []),
    _schema("end_combat", "Finaliza combate.", {}, []),

    _schema("start_turn", "Resetea movimiento del actor actual del combate.", {}, []),
    _schema("move", "Mueve un actor X pies (gasta move_left).", {"target": {"type": "string"}, "feet": {"type": "integer"}}, ["target", "feet"]),
    _schema("stand_up", "Levanta a un actor de PRONE gastando movimiento.", {"target": {"type": "string"}}, ["target"]),

    _schema("set_range", "Fija banda de distancia de un objetivo.", {"target": {"type": "string"}, "band": {"type": "string"}}, ["target", "band"]),
    _schema("get_range", "Devuelve banda de distancia entre A y B.", {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"]),
    _schema("approach", "Pone a actor en melee (simplificado).", {"actor": {"type": "string"}}, ["actor"]),
    _schema("retreat", "Aleja a actor a short/medium (simplificado).", {"actor": {"type": "string"}}, ["actor"]),

    _schema("apply_condition", "Aplica condición por X rondas.", {"target": {"type": "string"}, "condition": {"type": "string"}, "rounds": {"type": "integer"}}, ["target", "condition"]),
    _schema("remove_condition", "Quita condición.", {"target": {"type": "string"}, "condition": {"type": "string"}}, ["target", "condition"]),
    _schema("target_status", "Estado de un objetivo.", {"target": {"type": "string"}}, ["target"]),
    _schema("conditions_status", "Lista condiciones activas.", {}, []),

    _schema("heal", "Cura a un PJ.", {"target": {"type": "string"}, "amount": {"type": "integer"}}, ["target", "amount"]),
    _schema("damage", "Aplica daño a un PJ.", {"target": {"type": "string"}, "amount": {"type": "integer"}}, ["target", "amount"]),
    _schema("stabilize", "Estabiliza PJ a 0 HP.", {"target": {"type": "string"}}, ["target"]),
    _schema("rest", "Descanso short/long.", {"kind": {"type": "string"}}, []),

    _schema("attack", "Resuelve un ataque contra ENEMIGO.", {
        "attacker": {"type": "string"},
        "target": {"type": "string"},
        "bonus": {"type": "integer"},
        "damage": {"type": "string"},
        "attack_type": {"type": "string"},
        "power_shot": {"type": "boolean"},
        "power_attack": {"type": "boolean"},
        "auto_power": {"type": "boolean"},
        "crit_range": {"type": "integer"}
    }, ["attacker", "target", "bonus"]),
    _schema("bonus_attack", "Ejecuta el ataque extra si está disponible.", {
        "attacker": {"type": "string"},
        "target": {"type": "string"},
        "bonus": {"type": "integer"},
        "damage": {"type": "string"},
    }, ["attacker", "target", "bonus"]),

    _schema("xp_status", "Devuelve el estado de XP del grupo (total, por PJ, últimos logs).", {}, []),

    _schema("xp_progress", "Añade XP por progreso narrativo (minor/standard/major).", {
        "kind": {"type": "string"},
        "reason": {"type": "string"}
    }, ["kind"]),

    _schema("xp_kill", "Añade XP por enemigo muerto usando su CR desde canon.enemies.", {
        "enemy": {"type": "string"}
    }, ["enemy"]),

    _schema("level_status", "Devuelve nivel estimado por XP (y XP necesaria al siguiente).", {}, []),
    _schema("level_check_up", "Comprueba si el grupo sube de nivel según XP y actualiza canon.party.level (y members) si procede.", {}, []),
    _schema("level_up_announce", "Si procede, anuncia la subida de nivel por XP: 'Subís a nivel X'.", {}, []),

    _schema("memory_get", "Lee una clave de memoria.", {"key": {"type": "string"}}, ["key"]),
    _schema("memory_set", "Guarda en memoria (texto).", {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
]

# =========================================================
# Prompt / estilo DM
# =========================================================
def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"[No se pudo leer {path}: {e}]"

DM_STYLE = read_text_file(str(ROOT / "dm" / "style.md"))

SYSTEM = f"""
Eres un Dungeon Master (DM) para D&D 5e.
Prioridad: inmersión + continuidad + reglas consistentes.

ESTILO (OBLIGATORIO):
{DM_STYLE}

REGLAS DE USO:
- Si necesitas estado (HP, CA, condiciones, escena, combate), usa herramientas (canon_get / target_status / combat_status / etc.).
- Para mecánicas repetitivas, usa herramientas (roll/check/attack/move/apply_condition).
- Después de una resolución importante, registra 1–2 líneas con log_event().
- No inventes números si importan: consulta el canon.
- Para tiradas de habilidad de un PJ, usa skill_check(actor, skill, dc, mode) para calcular bonus automáticamente.

MODO ESCENA (CRÍTICO):
- dm_turn NO ES “solo combate”. Si no hay combate activo, SIEMPRE continúas la aventura en modo escena (exploración/social/decisiones).
- Nunca respondas “no hay combate activo” como bloqueo. Si no hay combate, describes la situación actual, propones 2–4 opciones y pides una decisión.
- Si falta contexto de escena, usa scene_status y canon_get para anclar continuidad antes de describir.

- REGLA CRÍTICA DE TRANSPARENCIA:
  Siempre que llames a una tool mecánica (roll/check/skill_check/attack/start_combat/combat_status/target_status),
  en tu siguiente mensaje DEBES incluir una sección "RESOLUCIÓN (mecánica)" y pegar el output de la tool literalmente
  (sin resumirlo). Después ya narras consecuencias.
- Cuando uses una herramienta (roll/check/skill_check/attack), pega el resultado literal en una sección ‘MECÁNICA:’ antes de narrar.
- Para cualquier conjuro (texto, alcance, duración, componentes, etc.), usa spell_info(name). Si dudas del nombre exacto, usa spell_search(query).
- Tras un hito importante, llama a xp_progress(...) y luego a level_check_up() para ver si subimos de nivel.
- Tras añadir XP, llama a level_up_announce() para anunciar la subida de nivel.

- MÓDULO (Fidelidad):
  - Si hay un módulo activo, ANTES de describir una escena nueva o resolver una decisión importante, llama a module_query()
    con palabras clave del momento (lugar, PNJ, objetivo, elemento raro) para anclar la narración al texto real.
  - Usa module_quote(chunk_id) SOLO cuando necesites texto literal (read-aloud) o verificación; evita soltar spoilers.
  - No menciones números de sala/encuentro ni claves internas; adapta la presentación al jugador.
  - Tras hitos grandes, actualiza module_set_progress(chapter, scene, flags) para mantener continuidad.

RECAP (OBLIGATORIO):
- Si has avanzado la escena o cambiado algo relevante, al final del turno llama a update_recap con un resumen de 2–6 líneas:
  lugar actual, qué acaba de pasar, NPCs relevantes, y qué decisiones quedan abiertas.
"""

# =========================================================
# Agente (Chat Completions + function calling)
# =========================================================
from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import time
import inspect

@dataclass
class AgentState:
    # Historial para el LLM (Chat Completions messages)
    history: List[dict] = field(default_factory=list)

    # Estado persistente "ligero" (no depende de releer todo history)
    scene: Dict[str, Any] = field(default_factory=lambda: {
        "location": "",
        "current_scene": "",
        "recap": "",
        "open_threads": [],
    })

    # Flags generales de campaña (decisiones, puertas abiertas, PNJ hostiles, etc.)
    flags: Dict[str, Any] = field(default_factory=dict)

    # Progreso del módulo (si se usa)
    module_progress: Dict[str, Any] = field(default_factory=lambda: {
        "module_id": "",
        "chapter": "",
        "scene": "",
        "flags": [],
    })

def _system_msg() -> dict:
    return {"role": "system", "content": SYSTEM}

def _user_msg(text: str) -> dict:
    return _user_item(text)

def _user_item(text: str) -> dict:
    return {"role": "user", "content": str(text or "")}

def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": str(text or "")}

def _tool_msg(tool_call_id: str, output: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": str(output)}

_ACTIVE_STATE: Optional["AgentState"] = None

def _get_state() -> Optional["AgentState"]:
    return _ACTIVE_STATE

def _schemas_to_chat_tools(tool_schemas: list) -> list:
    """
    Convierte tus TOOL_SCHEMAS (formato responses-ish) a formato Chat Completions:
      {"type":"function","function":{"name","description","parameters"}}
    """
    out = []
    for s in tool_schemas or []:
        if not isinstance(s, dict):
            continue
        if s.get("type") != "function":
            continue
        out.append({
            "type": "function",
            "function": {
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "parameters": (s.get("parameters") or {"type": "object", "properties": {}}),
            }
        })
    return out

CHAT_TOOLS = _schemas_to_chat_tools(TOOL_SCHEMAS)

def _classify_openai_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    resp = getattr(e, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    msg = str(e)

    try:
        s = int(status) if status is not None else None
    except Exception:
        s = None

    if s == 401:
        return "AUTH"
    if s == 403:
        return "FORBIDDEN"
    if s == 404:
        return "MODEL_NOT_FOUND"
    if s == 429:
        if "insufficient_quota" in msg:
            return "INSUFFICIENT_QUOTA"
        return "RATE_LIMIT"
    if s is not None and 500 <= s < 600:
        return "SERVER"
    if "insufficient_quota" in msg:
        return "INSUFFICIENT_QUOTA"
    return "OTHER"

def _sleep_backoff(attempt: int) -> None:
    time.sleep(2 * (attempt + 1))

def _call_chat_with_retries(messages: List[dict], *, max_attempts: int = 4):
    last_err: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(
                model=_require_model(),
                messages=messages,
                tools=CHAT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            last_err = e
            kind = _classify_openai_error(e)

            if kind == "INSUFFICIENT_QUOTA":
                raise RuntimeError("OPENAI_ERROR:INSUFFICIENT_QUOTA | La API key no tiene cuota/billing activo.") from e
            if kind == "AUTH":
                raise RuntimeError("OPENAI_ERROR:AUTH | API key inválida/no configurada (401).") from e
            if kind == "MODEL_NOT_FOUND":
                raise RuntimeError(f"OPENAI_ERROR:MODEL_NOT_FOUND | Modelo no disponible.") from e

            if kind in {"RATE_LIMIT", "SERVER"} and attempt < (max_attempts - 1):
                _sleep_backoff(attempt)
                continue

            if attempt < (max_attempts - 1):
                _sleep_backoff(attempt)
                continue

    raise RuntimeError(f"OPENAI_ERROR:OTHER | {last_err}") from last_err

def _parse_tool_args(args_raw: Any) -> dict:
    if isinstance(args_raw, dict):
        return args_raw
    try:
        obj = json.loads(args_raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _filter_args_to_signature(fn: Callable[..., Any], args: dict) -> Tuple[dict, List[str]]:
    dropped: List[str] = []
    try:
        allowed = set(inspect.signature(fn).parameters.keys())
        filtered = {}
        for k, v in (args or {}).items():
            if k in allowed:
                filtered[k] = v
            else:
                dropped.append(k)
        return filtered, dropped
    except Exception:
        return args or {}, dropped

def _normalize_history_for_chat(history: list) -> list:
    """
    Convierte historial viejo (Responses API) a formato Chat Completions.

    Fix crítico:
    - Conserva tool_calls en mensajes del assistant.
    - Elimina mensajes tool "huérfanos" (sin tool_calls previos) para evitar:
      "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'."
    """
    out = []
    pending_tool_ids = set()  # tool_call_ids esperados tras un assistant con tool_calls

    for m in history or []:
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        content = m.get("content")

        # Caso Responses: content = [{"type":"input_text","text":"..."}]
        if isinstance(content, list) and content:
            parts = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t in {"input_text", "text", "output_text"}:
                        parts.append(str(c.get("text") or ""))
            content = "\n".join([p for p in parts if p.strip()])

        # Assistant: conservar tool_calls si existen
        if role == "assistant":
            msg = {"role": "assistant", "content": str(content or "")}

            tool_calls = m.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                # Normaliza ids de tool_calls
                ids = set()
                norm_calls = []
                for tc in tool_calls:
                    if hasattr(tc, "model_dump"):
                        tc = tc.model_dump()
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        if tc_id:
                            ids.add(tc_id)
                        norm_calls.append(tc)
                msg["tool_calls"] = norm_calls
                pending_tool_ids = ids
            else:
                pending_tool_ids = set()

            out.append(msg)
            continue

        # Tool: Chat Completions requiere tool_call_id y debe corresponder a tool_calls previos
        if role == "tool":
            tool_call_id = (
                m.get("tool_call_id")
                or m.get("toolCallId")
                or m.get("call_id")
                or m.get("id")
            )
            if not tool_call_id:
                continue

            # Si no hay tool_calls previos o no coincide, es "huérfano" -> lo descartamos
            if not pending_tool_ids or tool_call_id not in pending_tool_ids:
                continue

            out.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(content or "")})
            pending_tool_ids.discard(tool_call_id)
            continue

        # system/user: normal
        out.append({"role": role, "content": str(content or "")})

    return out

import re
from typing import Tuple

# =========================================================
# Gameplay patch: menos literal (movimiento implícito, NPCs, aliases)
# =========================================================

# Aliases típicos (AD&D/ediciones viejas / nombres comunes)
_SPELL_ALIASES = {
    # classic/legacy -> 5e canonical spell names
    "cure light wounds": "cure wounds",
    "cure serious wounds": "cure wounds",
    "cause light wounds": "inflict wounds",
    "cause serious wounds": "inflict wounds",

    # common casing / spacing variants (safe)
    "magic missile": "magic missile",
    "fireball": "fireball",
    "hold person": "hold person",
    "faerie fire": "faerie fire",
    "lightning bolt": "lightning bolt",
}

def _apply_spell_aliases(text: str) -> Tuple[str, list]:
    """Normaliza nombres de spells usando aliases y devuelve (texto_nuevo, cambios)."""
    if not text:
        return text, []
    changes = []
    out = text

    for src, dst in _SPELL_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(dst, out)
            changes.append((src, dst))

    return out, changes

_TERM_ALIASES = {
    # Feats / shorthand
    "sharp shoot": "sharpshooter",
    "sharp shooter": "sharpshooter",
    "sharpshoot": "sharpshooter",
    "great weapon mastery": "great weapon master",
    "gwm": "great weapon master",
    "ss": "sharpshooter",

    # Combat terms
    "sneak attack": "sneak attack",
    "sa": "sneak attack",

    # Spanish -> canonical concept tokens (kept as terms)
    "sigilo": "stealth",
}

# Nombres que el usuario suele usar como “actor” aunque no esté en party
# (si no existe, lo tratamos como PNJ relevante por defecto)
_DEFAULT_ASSUME_NPC_NAMES = {"eldrin"}

def _apply_term_aliases(text: str) -> Tuple[str, list]:
    """Normaliza términos (feats, shorthand) y devuelve (texto_nuevo, cambios)."""
    if not text:
        return text, []
    changes = []
    out = text

    for src, dst in _TERM_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(dst, out)
            changes.append((src, dst))

    return out, changes

def _inject_intent_hints(text: str) -> str:
    """
    Intents: asunciones razonables para que el DM no sea literal.
    """
    t = text or ""
    tl = t.lower()

    # (A) Ataque a melé => moverse y atacar si es posible
    wants_melee = any(k in tl for k in ["a melé", "a melee", "melee", "cuerpo a cuerpo"])
    mentions_attack = any(k in tl for k in ["ataca", "ataque", "golpea", "carga", "embiste"])
    if wants_melee and mentions_attack:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA: si el atacante no está ya a melé, asume que se mueve lo necesario "
            "para ponerse a melé (usando su movimiento) y luego ejecuta el ataque en el mismo turno, "
            "salvo que sea físicamente imposible.]"
        )

    # (B) Sharpshooter / GWM => el usuario suele querer “usar el feat” automáticamente
    if "sharpshooter" in tl or "great weapon master" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA: si el usuario menciona Sharpshooter o Great Weapon Master, asume que "
            "quiere aplicar su opción de -5/+10 (si procede) y explica brevemente el impacto. "
            "Si no procede (por reglas/arma), indícalo sin bloquear.]"
        )

    # (C) Sneak Attack => si hay condiciones razonables, aplícalo sin que el usuario lo “pida perfecto”
    if "sneak attack" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA: si el usuario menciona Sneak Attack, aplica SA si se cumplen condiciones "
            "(ventaja, o aliado adyacente al objetivo, etc.). Si no se cumplen, dilo y sugiere cómo habilitarlo.]"
        )

    # (D) Sigilo/Stealth => el usuario quiere moverse con cautela y tirar stealth cuando haga falta
    if "stealth" in tl or "sigilo" in tl:
        t += (
            "\n\n[INTENCIÓN IMPLÍCITA: si el usuario indica sigilo/stealth, asume movimiento cauteloso y pide "
            "una tirada de Stealth solo cuando haya riesgo real de ser detectado. No bloquees por ello.]"
        )

    # (E) PNJs no registrados (Eldrin, etc.)
    for name in _DEFAULT_ASSUME_NPC_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", tl):
            t += (
                f"\n\n[NOTA: si '{name.title()}' no está en la party/canon, trátalo como PNJ aliado relevante "
                "en la escena (no bloquees). Si necesitas concretar stats/rol, pregunta 1 cosa concreta.]"
            )
            break

    return t

def _preprocess_user_text(user_text: str) -> str:
    """
    Normaliza input del usuario para hacerlo más jugable:
    - aliases de spells
    - aliases de términos (feats/shorthand/idioma)
    - hints de intención (mover y atacar, PNJs no registrados, etc.)
    """
    t = (user_text or "").strip()

    # 1) aliases de spells
    t2, spell_changes = _apply_spell_aliases(t)

    # 2) aliases de términos (feats/shorthand/idioma)
    t3, term_changes = _apply_term_aliases(t2)

    changes = []
    changes.extend(spell_changes)
    changes.extend(term_changes)

    if changes:
        t3 += "\n\n[ALIASES APLICADOS: " + ", ".join([f"'{a}'→'{b}'" for a, b in changes]) + "]"

    # 3) intención implícita
    t3 = _inject_intent_hints(t3)

    return t3

def run_agent_turn(user_text: str, state: AgentState) -> str:
    """
    Ejecuta un turno del agente.
    A3: activa un puntero global al AgentState actual para que las tools puedan
    persistir scene/location/flags/module_progress en la sesión (no solo en history).
    """
    global _ACTIVE_STATE
    _ACTIVE_STATE = state
    try:
        state.history = _normalize_history_for_chat(state.history)

        if not state.history or state.history[0].get("role") != "system":
            state.history.insert(0, _system_msg())

        user_text = _preprocess_user_text(user_text)
        state.history.append(_user_msg(user_text))


        try:
            resp = _call_chat_with_retries(state.history)
        except Exception as e:
            return f"Error llamando al modelo: {e}"

        while True:
            msg = resp.choices[0].message

            # 1) Si hay tool calls, ejecútalas
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                # guarda el mensaje del assistant tal cual (con tool_calls)
                state.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
                })

                for tc in tool_calls:
                    tc_id = tc.id
                    fn_name = tc.function.name
                    fn_args = _parse_tool_args(tc.function.arguments)

                    fn = TOOLS.get(fn_name)
                    if not fn:
                        out = f"Error: herramienta '{fn_name}' no existe en TOOLS."
                    else:
                        filtered_args, dropped = _filter_args_to_signature(fn, fn_args)
                        try:
                            out = fn(**filtered_args)
                            if dropped:
                                out = f"{out}\n(WARN: args ignorados por no estar en la firma: {dropped})"
                        except Exception as e:
                            out = f"Error ejecutando {fn_name}: {e}"

                    state.history.append(_tool_msg(tc_id, str(out)))

                # 2) vuelve a llamar al modelo con tool outputs ya en messages
                try:
                    resp = _call_chat_with_retries(state.history)
                except Exception as e:
                    return f"Error llamando al modelo tras tools: {e}"

                continue

            # 3) Sin tools: devuelve texto final
            text = (msg.content or "").strip()
            if text:
                # Si el modelo se “auto-bloquea” por combate, forzamos un retry 1 vez en modo escena
                if re.search(r"no hay un combate activo", text, re.IGNORECASE):
                    state.history.append(_assistant_msg(text))
                    state.history.append(_user_msg(
                        "Continúa en MODO ESCENA (sin combate): describe la situación actual, mantén continuidad, "
                        "da 2–4 opciones accionables. No te bloquees por falta de combate."
                    ))
                    try:
                        resp = _call_chat_with_retries(state.history)
                        msg2 = resp.choices[0].message
                        text2 = (msg2.content or "").strip()
                        if text2:
                            state.history.append(_assistant_msg(text2))
                            return text2
                    except Exception as e:
                        return f"Error llamando al modelo tras retry modo escena: {e}"

                state.history.append(_assistant_msg(text))
                return text

            return "(Sin salida de texto; revisa el prompt/tools.)"
    finally:
        _ACTIVE_STATE = None

# =========================================================
# CLI (robusto + modo local)
# =========================================================
if __name__ == "__main__":
    state = AgentState()
    print("DM Agent listo. Escribe 'exit' para salir. Usa '/help' para comandos locales.\n")

    def _local_help() -> str:
        return (
            "Comandos locales (sin LLM):\n"
            "  /help\n"
            "  /scene                 -> scene_status()\n"
            "  /combat                -> combat_status()\n"
            "  /status <nombre>       -> target_status(nombre)\n"
            "  /roll <expr>           -> roll(expr)\n"
            "  /check <skill> <dc> <bonus> [normal|adv|dis]\n"
            "  /skill <actor> <skill> <dc> [normal|adv|dis] [extra_bonus]\n"
            "  /spell <nombre>        -> spell_info(nombre)\n"
            "  /spells <query> [n]    -> spell_search(query, n)\n"
            "  /module_load            -> module_load()\n"
            "  /mfind <query>           -> module_query(query, 6)\n"
            "  /mquote <chunk_id>       -> module_quote(chunk_id)\n"
            "  /exit\n"
        )

    while True:
        try:
            text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.\n")
            break

        t = text.lower().strip()
        if t in {"exit", "quit", "salir", ":q", "/exit"}:
            print("Saliendo.\n")
            break

        # ---- modo local (sin modelo) ----
        if t in {"/help", "help"}:
            print(_local_help())
            continue

        if t in {"/scene"}:
            print("\nAgente:\n", tool_scene_status(), "\n")
            continue

        if t in {"/combat"}:
            print("\nAgente:\n", tool_combat_status(), "\n")
            continue

        if t.startswith("/status "):
            name = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_target_status(name), "\n")
            continue

        if t.startswith("/roll "):
            expr = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_roll(expr), "\n")
            continue

        if t.startswith("/spell "):
            name = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_spell_info(name), "\n")
            continue

        if t.startswith("/spells "):
            parts = text.split()
            query = parts[1] if len(parts) >= 2 else ""
            limit = int(parts[2]) if len(parts) >= 3 else 10
            print("\nAgente:\n", tool_spell_search(query, limit), "\n")
            continue

        if t.startswith("/check "):
            parts = text.split()
            # /check stealth 15 5 adv
            if len(parts) < 4:
                print("\nAgente:\n Uso: /check <skill> <dc> <bonus> [normal|adv|dis]\n")
                continue
            skill = parts[1]
            dc = int(parts[2])
            bonus = int(parts[3])
            mode = parts[4] if len(parts) >= 5 else "normal"
            print("\nAgente:\n", tool_check(skill, dc, bonus, mode), "\n")
            continue

        if t.startswith("/skill "):
            parts = text.split()
            # /skill "Myrmyr Lash" stealth 15 adv 0  (sin comillas: usa underscore o escribe sin espacios)
            if len(parts) < 4:
                print("\nAgente:\n Uso: /skill <actor> <skill> <dc> [normal|adv|dis] [extra_bonus]\n")
                continue
            actor = parts[1]
            skill = parts[2]
            dc = int(parts[3])
            mode = parts[4] if len(parts) >= 5 else "normal"
            extra = int(parts[5]) if len(parts) >= 6 else 0
            print("\nAgente:\n", tool_skill_check(actor, skill, dc, mode, extra), "\n")
            continue
        
        if t in {"/module_load"}:
            print("\nAgente:\n", tool_module_load(), "\n")
            continue

        if t.startswith("/mfind "):
            q = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_module_query(q, 6), "\n")
            continue

        if t.startswith("/mquote "):
            cid = text.split(" ", 1)[1].strip()
            print("\nAgente:\n", tool_module_quote(cid), "\n")
            continue

        # ---- modo normal (con LLM) ----
        out = run_agent_turn(text, state)
        print("\nAgente:\n", out, "\n")
