from __future__ import annotations

import os, json, time, uuid, re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File as UpFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, text

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .db import get_db, ENGINE
from .models import User, Thread, Message, File, FileText, FileChunk, AuditLog, Agent, AgentKnowledge, AgentLink, CostEvent
from .security import require_secret, new_salt, pbkdf2_hash, verify_password, mint_token, decode_token
from .extractors import extract_text
from .retrieval import keyword_retrieve

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

APP_VERSION = "2.0.9"
RAG_MODE = "keyword"

def new_id() -> str:
    return uuid.uuid4().hex

def now_ts() -> int:
    return int(time.time())

def cors_list() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]

def tenant_mode() -> str:
    return os.getenv("TENANT_MODE", "multi")

def default_tenant() -> str:
    return os.getenv("DEFAULT_TENANT", "public")

def admin_api_key() -> str:
    return os.getenv("ADMIN_API_KEY", "").strip()

def admin_emails() -> List[str]:
    raw = os.getenv("ADMIN_EMAILS", "").strip()
    if not raw:
        return []
    return [x.strip().lower() for x in raw.split(",") if x.strip()]

def enable_streaming() -> bool:
    return os.getenv("ENABLE_STREAMING", "0").strip() in ("1", "true", "True")


def get_linked_agent_ids(db: Session, org: str, source_agent_id: str) -> List[str]:
    rows = db.execute(
        select(AgentLink.target_agent_id).where(
            AgentLink.org_slug == org,
            AgentLink.source_agent_id == source_agent_id,
            AgentLink.enabled == True,
        )
    ).all()
    out: List[str] = []
    for r in rows:
        if r and r[0]:
            out.append(r[0])
    # de-dup keep order
    return list(dict.fromkeys(out))

def get_agent_file_ids(db: Session, org: str, agent_ids: List[str]) -> List[str]:
    if not agent_ids:
        return []
    rows = db.execute(
        select(AgentKnowledge.file_id).where(
            AgentKnowledge.org_slug == org,
            AgentKnowledge.enabled == True,
            AgentKnowledge.agent_id.in_(agent_ids),
        )
    ).all()
    return [r[0] for r in rows if r and r[0]]

def get_org(x_org_slug: Optional[str]) -> str:
    if tenant_mode() == "single":
        return default_tenant()
    return (x_org_slug or default_tenant()).strip() or default_tenant()


def ensure_core_agents(db: Session, org: str) -> None:
    """Ensure the 3 core agents exist for the org (solo-supervised setup).
    Creates: Orkio (CEO) [default], Chris (VP/CFO), Orion (CTO).
    Idempotent and safe to call frequently.
    """
    # Only create if missing by name match (case-insensitive)
    existing = {a.name.strip().lower(): a for a in db.execute(select(Agent).where(Agent.org_slug == org)).scalars().all()}
    now = now_ts()

    def upsert(name: str, description: str, system_prompt: str, is_default: bool = False):
        key = name.strip().lower()
        a = existing.get(key)
        if a:
            # Keep user's edits; only set default flag if requested and no other default exists
            if is_default and not a.is_default:
                has_default = db.execute(select(func.count()).select_from(Agent).where(Agent.org_slug == org, Agent.is_default == True)).scalar() or 0
                if has_default == 0:
                    a.is_default = True
                    a.updated_at = now
                    db.add(a)
                    db.commit()
            return

        a = Agent(
            id=new_id(),
            org_slug=org,
            name=name,
            description=description,
            system_prompt=system_prompt,
            model=os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o-mini"),
            temperature=str(os.getenv("DEFAULT_TEMPERATURE", "0.35")),
            rag_enabled=True,
            rag_top_k=6,
            is_default=is_default,
            created_at=now,
            updated_at=now,
        )
        db.add(a)
        db.commit()
        existing[key] = a

    orkio_prompt = """Você é Orkio, o Agente CEO da plataforma Orkio.
Sua função é atuar como orquestrador estratégico, garantindo que decisões estejam alinhadas com:
- objetivos de negócio
- governança
- eficiência operacional
- segurança e controle

Responsabilidades principais:
- Análise estratégica: traduzir pedidos vagos em objetivos claros e priorizados.
- Delegação inteligente: quando houver especialização (financeira/técnica), delegar explicitamente ao agente correto.
- Governança e controle: respeitar contratos, limites de atuação e regras da plataforma; não assumir fatos sem evidência.
- Comunicação clara: respostas objetivas, estruturadas e orientadas à decisão; explicitar incertezas.

Regras de conduta:
- Não inventar dados.
- Não acessar documentos fora do escopo autorizado.
- Não executar ações financeiras ou jurídicas diretamente.

Tom de voz: estratégico, calmo, confiável, executivo.
Objetivo final: servir ao negócio com controle, previsibilidade, transparência e propósito.
"""

    chris_prompt = """Você é Chris, o Agente CFO da plataforma Orkio.
Sua função é representar a visão financeira, de controle e de risco do negócio, especialmente em ambientes regulados.

Responsabilidades principais:
- Análise financeira: avaliar decisões pelo impacto em custos, orçamento e sustentabilidade.
- Previsibilidade: ajudar a prever despesas operacionais, especialmente relacionadas ao uso de IA.
- Governança e compliance: priorizar auditoria, rastreabilidade e controle de gastos; identificar riscos financeiros e operacionais.
- Suporte ao CEO: recomendar de forma clara e prudente; questionar decisões que gerem custos imprevisíveis.

Regras de conduta:
- Não tomar a decisão estratégica final (papel do CEO).
- Não assumir crescimento sem planejamento de custos.
- Sempre indicar premissas, riscos e lacunas de dados.

Tom de voz: analítico, prudente, direto e orientado a números.
Objetivo final: tornar a IA financeiramente previsível, auditável e sustentável.
"""

    orion_prompt = """Você é Orion, o CTO Virtual do Orkio.
Sua função é proteger a arquitetura, a segurança e a governança técnica da plataforma.
Você nunca sacrifica estabilidade por velocidade, nem segurança por conveniência.

Princípios imutáveis:
- Segurança nunca é opcional
- Estabilidade vem antes de feature
- Sem contrato técnico, não entra
- Nada em produção sem rollback claro
- O que não é observável, não é confiável

Formato padrão de resposta:
1) Diagnóstico técnico
2) Estado atual (baseline)
3) Opções viáveis
4) Recomendação técnica
5) Riscos
6) Plano incremental
7) Critérios de aceite
8) Rollback / reversibilidade

Delegação:
- Produto/decisão final → Orkio (CEO)
- Custos/infra/escala → Chris (CFO)
"""

    upsert(
        name="Orkio (CEO)",
        description="Clone estratégico. Orquestra agentes e consolida decisões com governança.",
        system_prompt=orkio_prompt,
        is_default=True,
    )
    upsert(
        name="Chris (VP/CFO)",
        description="Custos, risco e previsibilidade. Recomendações conservadoras e auditáveis.",
        system_prompt=chris_prompt,
        is_default=False,
    )
    upsert(
        name="Orion (CTO)",
        description="Arquitetura, segurança e estabilidade. Evolução incremental sem regressões.",
        system_prompt=orion_prompt,
        is_default=False,
    )

class RegisterIn(BaseModel):
    tenant: str = Field(default_tenant(), min_length=1)
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=6, max_length=256)

class LoginIn(BaseModel):
    tenant: str = Field(default_tenant(), min_length=1)
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class ThreadIn(BaseModel):
    title: str = Field(default="Nova conversa", min_length=1, max_length=200)

class ThreadUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: int

class ChatIn(BaseModel):
    thread_id: Optional[str] = None
    agent_id: Optional[str] = None
    message: str = Field(min_length=1)
    top_k: int = 6

class ChatOut(BaseModel):
    thread_id: str
    answer: str
    citations: List[Dict[str, Any]] = []

def audit(db: Session, org_slug: str, user_id: Optional[str], action: str, request_id: str, path: str, status_code: int, latency_ms: int, meta: Optional[Dict[str, Any]] = None):
    a = AuditLog(
        id=new_id(),
        org_slug=org_slug,
        user_id=user_id,
        action=action,
        meta=json.dumps(meta or {}, ensure_ascii=False),
        request_id=request_id,
        path=path,
        status_code=status_code,
        latency_ms=latency_ms,
        created_at=now_ts(),
    )
    db.add(a)
    db.commit()

def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
        if payload.get("role") != "admin" and payload.get("approved_at") is None:
            raise HTTPException(status_code=403, detail="User pending approval")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(payload: Dict[str, Any]) -> None:
    if payload.get("role") == "admin":
        return
    raise HTTPException(status_code=403, detail="Admin required")

def require_admin_key(x_admin_key: Optional[str]) -> None:
    k = admin_api_key()
    # ADMIN_API_KEY is optional; if not configured, key-auth cannot be used.
    if not k:
        raise HTTPException(status_code=401, detail="ADMIN_API_KEY not configured")
    if not x_admin_key or x_admin_key != k:
        raise HTTPException(status_code=401, detail="Invalid admin key")

def require_admin_access(
    authorization: Optional[str] = Header(default=None),
    x_admin_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Allow admin via JWT (role=admin) OR via X-Admin-Key."""
    # 1) JWT path
    if authorization and authorization.lower().startswith("bearer "):
        payload = get_current_user(authorization)
        if payload.get("role") == "admin":
            return payload
        raise HTTPException(status_code=403, detail="Admin required")

    # 2) Admin key path
    require_admin_key(x_admin_key)
    return {"role": "admin", "via": "admin_key"}


def db_ok() -> bool:
    """Return True if database connection is healthy."""
    if ENGINE is None:
        return False
    try:
        from sqlalchemy import text as _text
        with ENGINE.connect() as conn:
            conn.execute(_text("SELECT 1"))
        return True
    except Exception:
        return False


app = FastAPI(title="Orkio API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_list(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rag_fallback_recent_chunks(db: Session, org: str, file_ids: List[str], top_k: int = 6) -> List[Dict[str, Any]]:
    """Fallback: when keyword retrieval yields nothing, return early chunks from the most recent file."""
    if not file_ids:
        return []
    row = db.execute(
        select(File.id).where(File.org_slug == org, File.id.in_(file_ids)).order_by(File.created_at.desc()).limit(1)
    ).first()
    if not row or not row[0]:
        return []
    fid = row[0]
    chunks = db.execute(
        select(FileChunk).where(FileChunk.org_slug == org, FileChunk.file_id == fid).order_by(FileChunk.idx.asc()).limit(top_k)
    ).scalars().all()
    if not chunks:
        return []
    f = db.get(File, fid)
    filename = f.filename if f else fid
    out: List[Dict[str, Any]] = []
    for c in chunks:
        out.append({"file_id": fid, "filename": filename, "content": c.content, "score": 0.0, "idx": getattr(c, "idx", None), "fallback": True})
    return out


# --- Railway / Edge hardening: always answer CORS preflight ---
# Some proxies may return 502 if OPTIONS is not answered quickly.
# CORSMiddleware should handle it, but this catch-all guarantees a fast 204.
from fastapi import Response as _Resp

@app.options('/{path:path}')
async def _preflight(path: str):
    return _Resp(status_code=204)


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("x-request-id") or new_id()
    start = time.time()
    try:
        resp = await call_next(request)
    finally:
        pass
    resp.headers["x-request-id"] = rid
    resp.headers["x-orkio-version"] = APP_VERSION
    return resp

@app.on_event("startup")
def _startup():
    # Hard safety gate: JWT secret must exist.
    require_secret()

    # DB is optional for smoke tests, but if configured we ensure tables exist.
    if ENGINE is not None:
        try:
            from .db import Base  # type: ignore
            Base.metadata.create_all(bind=ENGINE)
        except Exception:
            # Never crash startup for auto-create issues; health endpoint will reveal DB status.
            pass

    # ADMIN_API_KEY is optional. If not set, admin access is granted only via admin-role JWT.
    # (ADMIN_EMAILS controls who becomes admin on register/login.)
    return None


@app.get("/")
def root():
    # Railway default healthcheck may hit "/"
    return {"status": "ok", "service": "orkio-api", "version": APP_VERSION}

@app.get("/health")
def health_root():
    return {"status": "ok", "service": "orkio-api", "version": APP_VERSION}

@app.get("/api/health")
def health():
    return {"status": "ok", "db": "ok" if db_ok() else "down", "version": APP_VERSION, "rag": RAG_MODE}

@app.get("/api/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/auth/register", response_model=TokenOut)
def register(inp: RegisterIn, db: Session = Depends(get_db)):
    org = (inp.tenant or default_tenant()).strip()
    email = inp.email.lower().strip()
    # auto-admin
    role = "admin" if email in admin_emails() else "user"

    existing = db.execute(select(User).where(User.org_slug == org, User.email == email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    salt = new_salt()
    pw_hash = pbkdf2_hash(inp.password, salt)
    u = User(id=new_id(), org_slug=org, email=email, name=inp.name.strip(), role=role, salt=salt, pw_hash=pw_hash, created_at=now_ts())
    db.add(u)
    db.commit()

    token = mint_token({"sub": u.id, "org": org, "email": u.email, "name": u.name, "role": u.role, "approved_at": getattr(u, "approved_at", None)})
    return {"access_token": token, "token_type": "bearer", "user": {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "approved_at": getattr(u, "approved_at", None)}}

@app.post("/api/auth/login", response_model=TokenOut)
def login(inp: LoginIn, db: Session = Depends(get_db)):
    org = (inp.tenant or default_tenant()).strip()
    email = inp.email.lower().strip()
    u = db.execute(select(User).where(User.org_slug == org, User.email == email)).scalar_one_or_none()
    if not u or not verify_password(inp.password, u.salt, u.pw_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = mint_token({"sub": u.id, "org": org, "email": u.email, "name": u.name, "role": u.role, "approved_at": getattr(u, "approved_at", None)})
    return {"access_token": token, "token_type": "bearer", "user": {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "approved_at": getattr(u, "approved_at", None)}}

@app.get("/api/threads")
def list_threads(x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)

    # Ensure core agents exist (solo-supervised defaults)
    ensure_core_agents(db, org)
    rows = db.execute(select(Thread).where(Thread.org_slug == org).order_by(Thread.created_at.desc())).scalars().all()
    return [{"id": t.id, "title": t.title, "created_at": t.created_at} for t in rows]

@app.post("/api/threads")
def create_thread(inp: ThreadIn, x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    t = Thread(id=new_id(), org_slug=org, title=inp.title, created_at=now_ts())
    db.add(t)
    db.commit()
    return {"id": t.id, "title": t.title, "created_at": t.created_at}

@app.patch("/api/threads/{thread_id}")
def rename_thread(thread_id: str, inp: ThreadUpdate, x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    t = db.execute(select(Thread).where(Thread.org_slug == org, Thread.id == thread_id)).scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    t.title = inp.title.strip()
    db.add(t)
    db.commit()
    return {"id": t.id, "title": t.title, "created_at": t.created_at}

@app.get("/api/messages")
def list_messages(thread_id: str, x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    rows = db.execute(select(Message).where(Message.org_slug == org, Message.thread_id == thread_id).order_by(Message.created_at.asc())).scalars().all()
    return [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at, "agent_id": getattr(m, "agent_id", None), "agent_name": getattr(m, "agent_name", None)} for m in rows]


def _openai_answer(
    user_message: str,
    context_chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Answer using OpenAI Chat Completions, with optional thread history.

    history: list of {role: 'user'|'assistant'|'system', content: str}
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = (model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
    if not key or OpenAI is None:
        return None
    client = OpenAI(api_key=key)

    # Build context string (RAG)
    ctx = ""
    for c in (context_chunks or [])[:6]:
        fn = c.get("filename") or c.get("file_id")
        ctx += f"\n\n[Arquivo: {fn}]\n{c.get('content','')}"

    system = system_prompt or "Você é o Orkio. Responda de forma objetiva. Use o contexto de documentos quando disponível."

    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system})

    # Provide RAG context in a separate system message (keeps user message clean)
    if ctx.strip():
        messages.append({"role": "system", "content": f"Contexto de documentos (evidências):\n{ctx}"})

    # Add conversation history (if any)
    if history:
        for h in history[-24:]:
            r = (h.get("role") or "").strip()
            c = (h.get("content") or "").strip()
            if not r or not c:
                continue
            if r not in ("user", "assistant", "system"):
                r = "user"
            messages.append({"role": r, "content": c})

    # Finally, current user message
    messages.append({"role": "user", "content": user_message})

    try:
        kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = client.chat.completions.create(**kwargs)
        return {"text": (r.choices[0].message.content or "").strip(), "usage": getattr(r, "usage", None), "model": model}
    except Exception:
        return None





@app.post("/api/chat", response_model=ChatOut)
def chat(
    inp: ChatIn,
    x_org_slug: Optional[str] = Header(default=None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    org = get_org(x_org_slug)

    # Ensure thread
    tid = inp.thread_id
    if not tid:
        t = Thread(id=new_id(), org_slug=org, title="Nova conversa", created_at=now_ts())
        db.add(t)
        db.commit()
        tid = t.id

    # Parse @mentions to support multi-agent collaboration in the same thread
    mention_tokens = []
    try:
        mention_tokens = re.findall(r"@([A-Za-z0-9_\-]{2,64})", inp.message or "")
        # de-dup preserve order
        seen = set()
        mention_tokens = [m for m in mention_tokens if not (m.lower() in seen or seen.add(m.lower()))]
    except Exception:
        mention_tokens = []

    # Expand @Time to core team agents (Orkio, Chris, Orion)
    has_team = any(m.strip().lower() in ("time", "team") for m in (mention_tokens or []))

    # Load all agents once (for mention resolution)
    all_agents = db.execute(select(Agent).where(Agent.org_slug == org)).scalars().all()
    alias_to_agent = {}
    for a in all_agents:
        if not a or not a.name:
            continue
        full = a.name.strip().lower()
        alias_to_agent[full] = a
        first = full.split()[0] if full.split() else full
        if first:
            alias_to_agent.setdefault(first, a)

    # Determine target agents (mentions > explicit agent_id > default agent)
    target_agents: List[Agent] = []
    if mention_tokens:
        for tok in mention_tokens:
            a = alias_to_agent.get(tok.strip().lower())
            if a:
                target_agents.append(a)

    if has_team:
        # Ensure the 3 core agents are included
        core_names = ["orkio (ceo)", "chris (vp/cfo)", "orion (cto)"]
        for cn in core_names:
            a = alias_to_agent.get(cn)
            if a:
                target_agents.append(a)

        # de-dup preserve order
        seen_ids=set()
        dedup=[]
        for a in target_agents:
            if a and a.id not in seen_ids:
                dedup.append(a)
                seen_ids.add(a.id)
        target_agents=dedup

    if not target_agents:
        # fallback to selected agent or default
        agent = None
        if inp.agent_id:
            agent = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == inp.agent_id)).scalar_one_or_none()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        else:
            agent = db.execute(select(Agent).where(Agent.org_slug == org, Agent.is_default == True)).scalar_one_or_none()
        if agent:
            target_agents = [agent]

    # Save user message
    m_user = Message(
        id=new_id(),
        org_slug=org,
        thread_id=tid,
        user_id=user.get("sub"),
        user_name=user.get("name"),
        role="user",
        content=inp.message,
        created_at=now_ts(),
    )
    db.add(m_user)
    db.commit()

    # Build shared thread history for context (agents can see each other's messages)
    prev = db.execute(
        select(Message)
        .where(Message.org_slug == org, Message.thread_id == tid, Message.id != m_user.id)
        .order_by(Message.created_at.asc())
    ).scalars().all()

    history: List[Dict[str, str]] = []
    # Keep only the last ~24 messages
    prev = prev[-24:]
    for pm in prev:
        role = "assistant" if pm.role == "assistant" else ("system" if pm.role == "system" else "user")
        content = pm.content or ""
        if role == "assistant" and pm.agent_name:
            content = f"[@{pm.agent_name}] {content}"
        history.append({"role": role, "content": content})

    # Determine top_k: use agent's rag_top_k if available, else input's top_k
    answers = []
    all_citations: List[Dict[str, Any]] = []

    # If no agent exists at all, still try to answer without system prompt
    if not target_agents:
        target_agents = [None]  # type: ignore

    for agent in target_agents:
        # Scoped knowledge (agent + linked agents) + thread-scoped temp files
        agent_file_ids: List[str] | None = None
        if agent:
            linked_agent_ids = get_linked_agent_ids(db, org, agent.id)
            scope_agent_ids = [agent.id] + linked_agent_ids
            agent_file_ids = get_agent_file_ids(db, org, scope_agent_ids)

            # Include thread-scoped temporary files (uploads with intent='chat')
            if tid:
                thread_file_ids = [
                    r[0]
                    for r in db.execute(
                        select(File.id).where(
                            File.org_slug == org,
                            File.scope_thread_id == tid,
                            File.origin == "chat",
                        )
                    ).all()
                ]
                if thread_file_ids:
                    agent_file_ids = list(dict.fromkeys((agent_file_ids or []) + thread_file_ids))

        effective_top_k = (agent.rag_top_k if agent and agent.rag_enabled else inp.top_k)

        citations: List[Dict[str, Any]] = []
        if (not agent) or agent.rag_enabled:
            citations = keyword_retrieve(db, org_slug=org, query=inp.message, top_k=effective_top_k, file_ids=agent_file_ids)

            # Fallback for summary-style requests
            if (not citations) and agent_file_ids:
                q = (inp.message or "").lower()
                if any(k in q for k in ["resumo", "resuma", "sumar", "summary", "sintet", "analis", "analise"]):
                    citations = rag_fallback_recent_chunks(db, org=org, file_ids=agent_file_ids, top_k=effective_top_k)

        # Determine temperature
        temperature = None
        if agent and agent.temperature:
            try:
                temperature = float(agent.temperature)
            except Exception:
                pass

        # If this was a mention-driven multi-agent call, guide the agent to answer in its role
        user_msg = inp.message
        if agent and mention_tokens:
            user_msg = (
                f"Você foi acionado como [@{agent.name}] em um chat multi-agente. "
                f"Responda de forma objetiva e útil dentro do seu papel.\n\n"
                f"Mensagem do usuário: {inp.message}"
            )

        ans_obj = _openai_answer(
            user_msg,
            citations,
            history=history,
            system_prompt=(agent.system_prompt if agent else None),
            model_override=(agent.model if agent else None),
            temperature=temperature,
        )
        answer = (ans_obj.get("text") if ans_obj else None)

        if not answer:
            if citations:
                snippet = (citations[0].get("content") or "")[:600]
                fn = citations[0].get("filename") or citations[0].get("file_id")
                answer = f"Encontrei esta informação no documento ({fn}):\n\n{snippet}"
            else:
                answer = "Ainda não encontrei informação nos documentos enviados para responder com precisão. Você pode anexar um documento relacionado?"

        # Save assistant message for this agent
        m_ass = Message(
            id=new_id(),
            org_slug=org,
            thread_id=tid,
            role="assistant",
            content=answer,
            agent_id=(agent.id if agent else None),
            agent_name=(agent.name if agent else None),
            created_at=now_ts(),
        )
        db.add(m_ass)
        db.commit()

        # Persist token costs (admin dashboard)
        try:
            usage = (ans_obj.get("usage") if ans_obj else None)
            if usage:
                db.add(CostEvent(
                    id=new_id(),
                    org_slug=org,
                    user_id=user.get("sub"),
                    thread_id=tid,
                    message_id=m_ass.id,
                    agent_id=(agent.id if agent else None),
                    model=(ans_obj.get("model") if ans_obj else None),
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                    created_at=now_ts(),
                ))
                db.commit()
        except Exception:
            pass

        if agent and len(target_agents) > 1:
            answers.append(f"[@{agent.name}] {answer}")
        else:
            answers.append(answer)

        # Keep citations from the first agent (or merge small)
        if citations and not all_citations:
            all_citations = citations


    # If this was a team/multi-agent run, add a final consolidation by Orkio (CEO)
    if len(target_agents) > 1 and (has_team or mention_tokens):
        orkio_agent = alias_to_agent.get("orkio (ceo)") or alias_to_agent.get("orkio")
        if orkio_agent:
            try:
                # Build a compact synthesis prompt using the agents' outputs
                parts = "\n\n".join([a for a in answers if a])
                synth_user = (
                    "Você é Orkio (CEO). Consolide as contribuições dos agentes em uma entrega final única, "
                    "bem estruturada e executável. Não repita tudo; sintetize. "
                    "Inclua: visão, objetivos, plano por etapas, riscos, checklist e próximo passo.\n\n"
                    f"Contribuições:\n{parts}"
                )
                synth_obj = _openai_answer(
                    synth_user,
                    citations=[],  # avoid injecting new RAG chunks here; just consolidate
                    history=history,
                    system_prompt=orkio_agent.system_prompt,
                    model_override=orkio_agent.model,
                    temperature=float(orkio_agent.temperature) if orkio_agent.temperature else 0.35,
                )
                synth = (synth_obj.get("text") if synth_obj else None)
                if synth:
                    m_synth = Message(
                        id=new_id(),
                        org_slug=org,
                        thread_id=tid,
                        role="assistant",
                        content=synth,
                        agent_id=orkio_agent.id,
                        agent_name=orkio_agent.name,
                        created_at=now_ts(),
                    )
                    db.add(m_synth)
                    db.commit()
                    try:
                        usage = (synth_obj.get("usage") if synth_obj else None)
                        if usage:
                            db.add(CostEvent(
                                id=new_id(),
                                org_slug=org,
                                user_id=user.get("sub"),
                                thread_id=tid,
                                message_id=m_synth.id,
                                agent_id=orkio_agent.id,
                                model=(synth_obj.get("model") if synth_obj else None),
                                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                                created_at=now_ts(),
                            ))
                            db.commit()
                    except Exception:
                        pass
                    answers.append(f"[@{orkio_agent.name} — Consolidação] {synth}")
            except Exception:
                # Never fail the chat due to consolidation
                pass

    combined = "\n\n---\n\n".join([a for a in answers if a])

    return {"thread_id": tid, "answer": combined, "citations": all_citations}


@app.post("/api/files/upload")
async def upload(
    file: UploadFile = UpFile(...),
    agent_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    intent: Optional[str] = Form(None),
    link_agent: bool = Form(True),
    x_agent_id: Optional[str] = Header(default=None, alias="X-Agent-Id"),
    x_org_slug: Optional[str] = Header(default=None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    org = get_org(x_org_slug)
    try:
        filename = file.filename or "upload"
        raw = await file.read()
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (max 10MB)")

        effective_intent = (intent or '').strip().lower() or ('agent' if (link_agent and resolved_agent_id) else 'chat')
        # Normalize intent
        if effective_intent == 'chat':
            # Chat intent is temporary; never link to agent knowledge
            link_agent = False

        if effective_intent not in ('chat','agent','institutional'):
            effective_intent = 'agent' if (link_agent and resolved_agent_id) else 'chat'

        is_institutional = (effective_intent == 'institutional')
        f = File(
            id=new_id(),
            org_slug=org,
            thread_id=thread_id if effective_intent == 'chat' else None,
            filename=filename,
            original_filename=filename,
            origin=effective_intent,
            scope_thread_id=thread_id if effective_intent == 'chat' else None,
            scope_agent_id=resolved_agent_id if effective_intent == 'agent' else None,
            mime_type=file.content_type,
            size_bytes=len(raw),
            content=raw,
            extraction_failed=False,
            is_institutional=is_institutional,
            created_at=now_ts(),
        )
        db.add(f)
        db.commit()

        # Optionally link this upload to an agent (so RAG is scoped correctly)
        if link_agent and resolved_agent_id:
            try:
                ag = db.get(Agent, resolved_agent_id)
                if ag and ag.org_slug == org:
                    existing_link = db.execute(
                        select(AgentKnowledge).where(
                            AgentKnowledge.org_slug == org,
                            AgentKnowledge.agent_id == ag.id,
                            AgentKnowledge.file_id == f.id,
                        )
                    ).scalar_one_or_none()
                    if not existing_link:
                        db.add(
                            AgentKnowledge(
                                id=new_id(),
                                org_slug=org,
                                agent_id=ag.id,
                                file_id=f.id,
                                enabled=True,
                                created_at=now_ts(),
                            )
                        )
                        db.commit()
            except Exception:
                # Never fail upload due to linking
                pass

        extracted_chars = 0
        text_content = ""
        try:
            text_content, extracted_chars = extract_text(filename, raw)
            ft = FileText(id=new_id(), org_slug=org, file_id=f.id, text=text_content, extracted_chars=extracted_chars, created_at=now_ts())
            db.add(ft)

            # Chunking (deterministic)
            chunk_chars = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
            overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
            text_len = len(text_content)
            idx = 0
            pos = 0
            while pos < text_len:
                end = min(text_len, pos + chunk_chars)
                chunk = text_content[pos:end].strip()
                if chunk:
                    db.add(FileChunk(id=new_id(), org_slug=org, file_id=f.id, idx=idx, content=chunk, created_at=now_ts()))
                    idx += 1
                if end >= text_len:
                    break
                pos = max(0, end - overlap)

            db.commit()
        except Exception:
            f.extraction_failed = True
            db.add(f)
            db.commit()

        return {"file_id": f.id, "filename": f.filename, "status": "stored", "extracted_chars": extracted_chars}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"upload_failed: {e.__class__.__name__}: {str(e)}")

@app.get("/api/files")
def list_files(x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    rows = db.execute(select(File).where(File.org_slug == org).order_by(File.created_at.desc())).scalars().all()
    return [{"id": f.id, "filename": f.filename, "size_bytes": f.size_bytes, "extraction_failed": f.extraction_failed, "created_at": f.created_at} for f in rows]

# --- Admin ---
@app.get("/api/admin/overview")
def admin_overview(_admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    return {
        "tenants": db.execute(select(func.count(func.distinct(User.org_slug)))).scalar_one(),
        "users": db.execute(select(func.count(User.id))).scalar_one(),
        "threads": db.execute(select(func.count(Thread.id))).scalar_one(),
        "messages": db.execute(select(func.count(Message.id))).scalar_one(),
        "files": db.execute(select(func.count(File.id))).scalar_one(),
    }

@app.get("/api/admin/users")
def admin_users(status: str = "all", _admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    q = select(User)
    if status == "pending":
        q = q.where(User.approved_at == None)  # noqa: E711
    elif status == "approved":
        q = q.where(User.approved_at != None)  # noqa: E711
    rows = db.execute(q.order_by(User.created_at.desc()).limit(400)).scalars().all()
    return [{
        "id": u.id,
        "org_slug": u.org_slug,
        "email": u.email,
        "name": u.name,
        "role": u.role,
        "created_at": u.created_at,
        "approved_at": getattr(u, "approved_at", None),
    } for u in rows]

@app.post("/api/admin/users/{user_id}/approve")
def admin_approve_user(user_id: str, _admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    u = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    u.approved_at = now_ts()
    db.add(u)
    db.commit()
    return {"ok": True, "id": u.id, "approved_at": u.approved_at}

@app.post("/api/admin/users/{user_id}/reject")
def admin_reject_user(user_id: str, _admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    u = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(u)
    db.commit()
    return {"ok": True}
@app.get("/api/admin/files")
def admin_files(institutional_only: bool = False, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    q = select(File).where(File.org_slug == org)
    if institutional_only:
        q = q.where(File.is_institutional == True)
    rows = db.execute(q.order_by(File.created_at.desc()).limit(200)).scalars().all()
    return [{"id": f.id, "org_slug": f.org_slug, "filename": f.filename, "size_bytes": f.size_bytes, "extraction_failed": f.extraction_failed, "is_institutional": getattr(f, "is_institutional", False), "created_at": f.created_at} for f in rows]



@app.get("/api/admin/costs")
def admin_costs(days: int = 7, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    days = max(1, min(int(days or 7), 90))
    since = now_ts() - (days * 86400)

    rows = db.execute(
        select(
            CostEvent.agent_id,
            func.sum(CostEvent.total_tokens).label("total_tokens"),
            func.sum(CostEvent.prompt_tokens).label("prompt_tokens"),
            func.sum(CostEvent.completion_tokens).label("completion_tokens"),
        ).where(
            CostEvent.org_slug == org,
            CostEvent.created_at >= since,
        ).group_by(CostEvent.agent_id)
    ).all()

    total = db.execute(
        select(
            func.sum(CostEvent.total_tokens),
            func.sum(CostEvent.prompt_tokens),
            func.sum(CostEvent.completion_tokens),
        ).where(CostEvent.org_slug == org, CostEvent.created_at >= since)
    ).first()

    # Map agent_id -> name
    agent_map = {a.id: a.name for a in db.execute(select(Agent).where(Agent.org_slug == org)).scalars().all()}
    per_agent = []
    for r in rows:
        aid = r[0]
        per_agent.append({
            "agent_id": aid,
            "agent_name": agent_map.get(aid, "N/A") if aid else "N/A",
            "total_tokens": int(r[1] or 0),
            "prompt_tokens": int(r[2] or 0),
            "completion_tokens": int(r[3] or 0),
        })

    return {
        "org_slug": org,
        "days": days,
        "since": since,
        "total": {
            "total_tokens": int((total[0] or 0) if total else 0),
            "prompt_tokens": int((total[1] or 0) if total else 0),
            "completion_tokens": int((total[2] or 0) if total else 0),
        },
        "per_agent": sorted(per_agent, key=lambda x: x["total_tokens"], reverse=True),
    }
@app.post("/api/admin/files/upload")
async def admin_upload_file(file: UploadFile = UpFile(...), x_org_slug: Optional[str] = Header(default=None), _admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    """
    Upload institutional document (global) that can be linked to multiple agents.
    It is NOT auto-linked to any agent.
    """
    org = get_org(x_org_slug)
    filename = file.filename or "upload"
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Arquivo muito grande (max 10MB)")

    f = File(
        id=new_id(),
        org_slug=org,
        thread_id=None,
        filename=filename,
        mime_type=file.content_type,
        size_bytes=len(raw),
        content=raw,
        extraction_failed=False,
        is_institutional=True,
        created_at=now_ts(),
    )
    db.add(f)
    db.commit()

    extracted_chars = 0
    text_content = ""
    try:
        text_content, extracted_chars = extract_text(filename, raw)
        ft = FileText(id=new_id(), org_slug=org, file_id=f.id, text=text_content, extracted_chars=extracted_chars, created_at=now_ts())
        db.add(ft)

        # Chunking (deterministic)
        chunk_chars = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
        overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        text_len = len(text_content)
        idx = 0
        pos = 0
        while pos < text_len:
            end = min(text_len, pos + chunk_chars)
            chunk = text_content[pos:end].strip()
            if chunk:
                db.add(FileChunk(id=new_id(), org_slug=org, file_id=f.id, idx=idx, content=chunk, created_at=now_ts()))
                idx += 1
            if end >= text_len:
                break
            pos = max(0, end - overlap)

        db.commit()
    except Exception:
        f.extraction_failed = True
        db.add(f)
        db.commit()

    # audit
    try:
        audit(db, org_slug=org, user_id=None, action="admin_file_upload", request_id="admin", path="/api/admin/files/upload", status_code=200, latency_ms=0, meta={"file_id": f.id, "filename": f.filename, "is_institutional": True})
    except Exception:
        pass

    return {"file_id": f.id, "filename": f.filename, "status": "stored", "is_institutional": True, "extracted_chars": extracted_chars}

@app.get("/api/admin/audit")
def admin_audit(_admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    rows = db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(200)).scalars().all()
    out = []
    for a in rows:
        try:
            meta = json.loads(a.meta) if a.meta else {}
        except Exception:
            meta = {}
        out.append(
            {
                "id": a.id,
                "org_slug": a.org_slug,
                "user_id": a.user_id,
                "action": a.action,
                "meta": meta,
                "request_id": a.request_id,
                "path": a.path,
                "status_code": a.status_code,
                "latency_ms": a.latency_ms,
                "created_at": a.created_at,
            }
        )
    return out


class AgentIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=400)
    system_prompt: str = Field(default="", max_length=20000)
    model: Optional[str] = Field(default=None, max_length=80)
    embedding_model: Optional[str] = Field(default=None, max_length=80)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    rag_enabled: bool = True
    rag_top_k: int = Field(default=6, ge=1, le=20)
    is_default: bool = False

class AgentLinkIn(BaseModel):
    file_id: str
    enabled: bool = True

class AgentToAgentLinkIn(BaseModel):
    target_agent_ids: List[str] = Field(default_factory=list)
    mode: str = Field(default="consult")  # consult|delegate

class DelegateIn(BaseModel):
    source_agent_id: str = Field(min_length=1)
    target_agent_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=8000)
    create_thread: bool = True
    thread_title: Optional[str] = None



@app.post("/api/agents/delegate")
def agent_delegate(inp: DelegateIn, x_org_slug: Optional[str] = Header(default=None), _admin=Depends(require_admin_access), db: Session = Depends(get_db)):
    """Send a one-way instruction from source agent to target agent. Requires AgentLink(mode='delegate') enabled."""
    org = get_org(x_org_slug)

    source_agent_id = (inp.source_agent_id or "").strip()
    target_agent_id = (inp.target_agent_id or "").strip()
    if not source_agent_id or not target_agent_id:
        raise HTTPException(status_code=400, detail="source_agent_id and target_agent_id required")

    src = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == source_agent_id)).scalar_one_or_none()
    if not src:
        raise HTTPException(status_code=404, detail="Source agent not found")
    tgt = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == target_agent_id)).scalar_one_or_none()
    if not tgt:
        raise HTTPException(status_code=404, detail="Target agent not found")

    link = db.execute(
        select(AgentLink).where(
            AgentLink.org_slug == org,
            AgentLink.source_agent_id == source_agent_id,
            AgentLink.target_agent_id == target_agent_id,
            AgentLink.enabled == True,
            AgentLink.mode == "delegate",
        )
    ).scalar_one_or_none()
    if not link:
        raise HTTPException(status_code=403, detail="No delegate link from source to target")

    tid = None
    if inp.create_thread:
        title = (inp.thread_title or f"Instrução de {source_agent_id}").strip()[:200]
        t = Thread(id=new_id(), org_slug=org, title=title, created_at=now_ts())
        db.add(t)
        db.commit()
        tid = t.id

    sys_msg = Message(id=new_id(), org_slug=org, thread_id=tid, role="system", content=f"[delegate] source_agent_id={source_agent_id}", created_at=now_ts())
    usr_msg = Message(id=new_id(), org_slug=org, thread_id=tid, role="user", content=inp.instruction, created_at=now_ts())
    db.add(sys_msg); db.add(usr_msg); db.commit()

    citations: List[Dict[str, Any]] = []
    if tgt and tgt.rag_enabled:
        agent_file_ids = get_agent_file_ids(db, org, [target_agent_id])
        citations = keyword_retrieve(db, org_slug=org, query=inp.instruction, top_k=int(tgt.rag_top_k or 6), file_ids=agent_file_ids)

    answer = _openai_answer(
        inp.instruction,
        citations,
        system_prompt=tgt.system_prompt if tgt else None,
        model_override=tgt.model if tgt else None,
        temperature=float(tgt.temperature) if (tgt and tgt.temperature is not None) else None,
    ) or "Recebido. Vou seguir as orientações."

    ass_msg = Message(id=new_id(), org_slug=org, thread_id=tid, role="assistant", content=answer, agent_id=agent.id if agent else None, agent_name=agent.name if agent else None, created_at=now_ts())
    db.add(ass_msg); db.commit()

    try:
        audit(db, org_slug=org, user_id=None, action="agent_delegate", request_id="delegate", path="/api/agents/delegate", status_code=200, latency_ms=0, meta={"source_agent_id": source_agent_id, "target_agent_id": target_agent_id})
    except Exception:
        pass

    return {"ok": True, "thread_id": tid, "answer": answer, "citations": citations}

@app.get("/api/agents")
def list_agents(x_org_slug: Optional[str] = Header(default=None), user=Depends(get_current_user), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    rows = db.execute(select(Agent).where(Agent.org_slug == org).order_by(Agent.updated_at.desc())).scalars().all()
    return [{"id": a.id, "name": a.name, "description": a.description, "rag_enabled": a.rag_enabled, "rag_top_k": a.rag_top_k, "model": a.model, "temperature": a.temperature, "is_default": a.is_default, "updated_at": a.updated_at} for a in rows]



@app.get("/api/admin/agents/{agent_id}/links")
def admin_get_agent_links(agent_id: str, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    rows = db.execute(
        select(AgentLink).where(
            AgentLink.org_slug == org,
            AgentLink.source_agent_id == agent_id,
            AgentLink.enabled == True,
        ).order_by(AgentLink.created_at.desc())
    ).scalars().all()
    return [{"id": r.id, "source_agent_id": r.source_agent_id, "target_agent_id": r.target_agent_id, "mode": r.mode, "enabled": r.enabled, "created_at": r.created_at} for r in rows]

@app.put("/api/admin/agents/{agent_id}/links")
def admin_put_agent_links(agent_id: str, inp: AgentToAgentLinkIn, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    # ensure agent exists
    src = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == agent_id)).scalar_one_or_none()
    if not src:
        raise HTTPException(status_code=404, detail="Agent not found")

    # disable existing links
    existing = db.execute(select(AgentLink).where(AgentLink.org_slug == org, AgentLink.source_agent_id == agent_id)).scalars().all()
    for e in existing:
        e.enabled = False
        db.add(e)

    # validate targets (same org)
    targets: List[str] = []
    if inp.target_agent_ids:
        targets = db.execute(select(Agent.id).where(Agent.org_slug == org, Agent.id.in_(inp.target_agent_ids))).scalars().all()

    mode = (inp.mode or "consult").strip() or "consult"
    count = 0
    for tid in targets:
        if tid == agent_id:
            continue
        db.add(AgentLink(id=new_id(), org_slug=org, source_agent_id=agent_id, target_agent_id=tid, mode=mode, enabled=True, created_at=now_ts()))
        count += 1

    db.commit()
    return {"ok": True, "count": count}

@app.get("/api/admin/agents")
def admin_agents(_admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    # Admin can list per-org (from header) or all if header omitted in single-tenant mode
    org = get_org(x_org_slug)
    rows = db.execute(select(Agent).where(Agent.org_slug == org).order_by(Agent.updated_at.desc()).limit(200)).scalars().all()
    return [{"id": a.id, "org_slug": a.org_slug, "name": a.name, "description": a.description, "system_prompt": a.system_prompt, "rag_enabled": a.rag_enabled, "rag_top_k": a.rag_top_k, "model": a.model, "embedding_model": a.embedding_model, "temperature": a.temperature, "is_default": a.is_default, "created_at": a.created_at, "updated_at": a.updated_at} for a in rows]

@app.post("/api/admin/agents")
def admin_create_agent(inp: AgentIn, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    now = now_ts()
    # If setting as default, unset other defaults first
    if inp.is_default:
        db.execute(text("UPDATE agents SET is_default=0 WHERE org_slug=:org"), {"org": org})
    a = Agent(
        id=new_id(),
        org_slug=org,
        name=inp.name.strip(),
        description=inp.description,
        system_prompt=inp.system_prompt,
        model=inp.model,
        embedding_model=inp.embedding_model,
        temperature=str(inp.temperature) if inp.temperature is not None else None,
        rag_enabled=bool(inp.rag_enabled),
        rag_top_k=inp.rag_top_k,
        is_default=bool(inp.is_default),
        created_at=now,
        updated_at=now,
    )
    db.add(a)
    db.commit()
    return {"id": a.id}

@app.put("/api/admin/agents/{agent_id}")
def admin_update_agent(agent_id: str, inp: AgentIn, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    a = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == agent_id)).scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found")
    # If setting as default, unset other defaults first
    if inp.is_default and not a.is_default:
        db.execute(text("UPDATE agents SET is_default=0 WHERE org_slug=:org"), {"org": org})
    a.name = inp.name.strip()
    a.description = inp.description
    a.system_prompt = inp.system_prompt
    a.model = inp.model
    a.embedding_model = inp.embedding_model
    a.temperature = str(inp.temperature) if inp.temperature is not None else None
    a.rag_enabled = bool(inp.rag_enabled)
    a.rag_top_k = inp.rag_top_k
    a.is_default = bool(inp.is_default)
    a.updated_at = now_ts()
    db.add(a)
    db.commit()
    return {"ok": True}

@app.delete("/api/admin/agents/{agent_id}")
def admin_delete_agent(agent_id: str, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    a = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == agent_id)).scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found")
    db.execute(text("DELETE FROM agent_knowledge WHERE org_slug=:org AND agent_id=:aid"), {"org": org, "aid": agent_id})
    db.delete(a)
    db.commit()
    return {"ok": True}

@app.get("/api/admin/agents/{agent_id}/knowledge")
def admin_agent_knowledge(agent_id: str, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    rows = db.execute(select(AgentKnowledge).where(AgentKnowledge.org_slug == org, AgentKnowledge.agent_id == agent_id).order_by(AgentKnowledge.created_at.desc())).scalars().all()
    return [{"id": r.id, "file_id": r.file_id, "enabled": r.enabled, "created_at": r.created_at} for r in rows]

@app.post("/api/admin/agents/{agent_id}/knowledge")
def admin_add_agent_knowledge(agent_id: str, inp: AgentLinkIn, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    # ensure agent exists
    a = db.execute(select(Agent).where(Agent.org_slug == org, Agent.id == agent_id)).scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found")
    # upsert
    existing = db.execute(select(AgentKnowledge).where(AgentKnowledge.org_slug == org, AgentKnowledge.agent_id == agent_id, AgentKnowledge.file_id == inp.file_id)).scalar_one_or_none()
    if existing:
        existing.enabled = bool(inp.enabled)
        db.add(existing)
        db.commit()
        return {"id": existing.id}
    r = AgentKnowledge(id=new_id(), org_slug=org, agent_id=agent_id, file_id=inp.file_id, enabled=bool(inp.enabled), created_at=now_ts())
    db.add(r)
    db.commit()
    return {"id": r.id}

@app.delete("/api/admin/agents/{agent_id}/knowledge/{link_id}")
def admin_remove_agent_knowledge(agent_id: str, link_id: str, _admin=Depends(require_admin_access), x_org_slug: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    org = get_org(x_org_slug)
    r = db.execute(select(AgentKnowledge).where(AgentKnowledge.org_slug == org, AgentKnowledge.agent_id == agent_id, AgentKnowledge.id == link_id)).scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Link not found")
    db.delete(r)
    db.commit()
    return {"ok": True}
