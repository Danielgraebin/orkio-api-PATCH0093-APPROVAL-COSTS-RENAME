from __future__ import annotations
from sqlalchemy import Column, String, Text, BigInteger, Integer, LargeBinary, Boolean
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    email = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")  # user|admin
    salt = Column(String, nullable=False)
    pw_hash = Column(String, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    approved_at = Column(BigInteger, nullable=True)

class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(BigInteger, nullable=False)

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    thread_id = Column(String, index=True, nullable=False)
    user_id = Column(String, nullable=True)
    user_name = Column(String, nullable=True)
    role = Column(String, nullable=False)  # user|assistant|system
    content = Column(Text, nullable=False)
    agent_id = Column(String, nullable=True)
    agent_name = Column(String, nullable=True)
    created_at = Column(BigInteger, nullable=False)

class File(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    thread_id = Column(String, index=True, nullable=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)
    origin = Column(String, nullable=False, default='unknown')  # chat|agent|institutional
    scope_thread_id = Column(String, nullable=True)
    scope_agent_id = Column(String, nullable=True)
    mime_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=False, default=0)
    content = Column(LargeBinary, nullable=True)  # optional (MVP)
    extraction_failed = Column(Boolean, nullable=False, default=False)
    is_institutional = Column(Boolean, nullable=False, default=False)
    created_at = Column(BigInteger, nullable=False)

class FileText(Base):
    __tablename__ = "file_texts"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    file_id = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
    extracted_chars = Column(Integer, nullable=False, default=0)
    created_at = Column(BigInteger, nullable=False)

class FileChunk(Base):
    __tablename__ = "file_chunks"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    file_id = Column(String, index=True, nullable=False)
    idx = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    agent_id = Column(String, nullable=True)
    agent_name = Column(String, nullable=True)
    created_at = Column(BigInteger, nullable=False)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    user_id = Column(String, nullable=True)
    action = Column(String, nullable=False)
    meta = Column(Text, nullable=True)
    request_id = Column(String, nullable=True)
    path = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(BigInteger, nullable=False)


class Agent(Base):
    __tablename__ = "agents"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False, default="")
    model = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    temperature = Column(String, nullable=True)  # stored as string for flexibility
    rag_enabled = Column(Boolean, nullable=False, default=True)
    rag_top_k = Column(Integer, nullable=False, default=6)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

class AgentKnowledge(Base):
    __tablename__ = "agent_knowledge"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    agent_id = Column(String, index=True, nullable=False)
    file_id = Column(String, index=True, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(BigInteger, nullable=False)


class AgentLink(Base):
    __tablename__ = "agent_links"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    source_agent_id = Column(String, index=True, nullable=False)
    target_agent_id = Column(String, index=True, nullable=False)
    mode = Column(String, nullable=False, default="consult")  # consult|delegate
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(BigInteger, nullable=False)


class CostEvent(Base):
    __tablename__ = "cost_events"
    id = Column(String, primary_key=True)
    org_slug = Column(String, index=True, nullable=False)
    user_id = Column(String, nullable=True)
    thread_id = Column(String, index=True, nullable=True)
    message_id = Column(String, index=True, nullable=True)
    agent_id = Column(String, index=True, nullable=True)
    model = Column(String, nullable=True)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    created_at = Column(BigInteger, nullable=False)
