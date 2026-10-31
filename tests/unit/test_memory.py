"""Unit tests for the SQLite-backed conversation memory (no Ollama needed)."""

from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.ai.memory import ConversationMemory, MemoryConfig
from app.core.exceptions import (
    InvalidMessageError,
    SessionNotFoundError,
)
from app.db.database import create_db_engine, init_db
from app.db.models import DEFAULT_SESSION_TITLE, MessageRole

SUMMARY_TEXT = "ملخص تجريبي"


class FakeLLM:
    """Stands in for ChatOllama: records calls, optionally fails."""

    def __init__(self, reply: str = SUMMARY_TEXT, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return SimpleNamespace(content=self.reply)


@pytest.fixture()
def db():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()


def make_memory(db, llm=None) -> ConversationMemory:
    # 50 tokens is the smallest buffer; compaction starts above ~66 tokens.
    config = MemoryConfig(max_buffer_tokens=50, cross_session_limit=3)
    return ConversationMemory(db, llm or FakeLLM(), config=config)


def fill(memory: ConversationMemory, session_id: str, pairs: int) -> None:
    """Add alternating user/assistant messages of ~20 estimated tokens each."""
    for i in range(pairs):
        memory.add_message(session_id, MessageRole.USER, f"question {i} " + "x" * 50)
        memory.add_message(session_id, MessageRole.ASSISTANT, f"answer {i} " + "y" * 50)


def test_rejects_system_role_and_empty_content(db):
    memory = make_memory(db)
    session = memory.start_session()
    with pytest.raises(InvalidMessageError):
        memory.add_message(session.id, MessageRole.SYSTEM, "hack")
    with pytest.raises(InvalidMessageError):
        memory.add_message(session.id, MessageRole.USER, "   ")


def test_unknown_session_raises(db):
    with pytest.raises(SessionNotFoundError):
        make_memory(db).load_context("does-not-exist")


def test_first_user_message_becomes_title(db):
    memory = make_memory(db)
    session = memory.start_session()
    assert session.title == DEFAULT_SESSION_TITLE
    memory.add_message(session.id, MessageRole.USER, "كيف أزيد مبيعات متجري؟")
    assert memory.get_session(session.id).title == "كيف أزيد مبيعات متجري؟"


def test_short_conversation_needs_no_summary(db):
    llm = FakeLLM()
    memory = make_memory(db, llm)
    session = memory.start_session()
    memory.add_message(session.id, MessageRole.USER, "hello")
    memory.add_message(session.id, MessageRole.ASSISTANT, "hi")

    assert memory.compact(session.id) is False
    context = memory.load_context(session.id)
    assert len(context.recent_messages) == 2
    assert context.session_summary == ""
    assert llm.calls == 0


def test_compaction_summarises_overflow_and_keeps_history(db):
    memory = make_memory(db)
    session = memory.start_session()
    fill(memory, session.id, pairs=5)  # 10 messages, ~200 tokens

    assert memory.compact(session.id) is True
    context = memory.load_context(session.id)

    assert context.session_summary == SUMMARY_TEXT
    assert 1 <= len(context.recent_messages) <= 2
    assert context.recent_messages[0].role == "user"  # buffer starts on a user turn
    assert context.recent_messages[-1].content.startswith("answer 4")
    # Nothing is deleted, and the summary row never leaks into visible history.
    history = memory.get_history(session.id)
    assert len(history) == 10
    assert all(m.role != MessageRole.SYSTEM for m in history)


def test_summarisation_failure_is_not_fatal(db):
    memory = make_memory(db, FakeLLM(fail=True))
    session = memory.start_session()
    fill(memory, session.id, pairs=5)

    assert memory.compact(session.id) is False
    context = memory.load_context(session.id)
    assert context.session_summary == ""
    assert context.pending_unsummarised > 0  # retried on the next compaction


def test_reasoning_tags_are_removed_from_summary(db):
    memory = make_memory(db, FakeLLM(reply="<think>hidden</think>ملخص نظيف"))
    session = memory.start_session()
    fill(memory, session.id, pairs=5)
    memory.compact(session.id)
    assert memory.load_context(session.id).session_summary == "ملخص نظيف"


def test_new_session_loads_previous_session_digest(db):
    llm = FakeLLM()
    memory = make_memory(db, llm)
    old = memory.start_session()
    memory.add_message(old.id, MessageRole.USER, "أدير مطعمًا صغيرًا في الرياض")
    memory.add_message(old.id, MessageRole.ASSISTANT, "ما حجم الإيرادات الشهرية؟")

    new = memory.start_session()
    context = memory.load_context(new.id)

    assert [p.session_id for p in context.past_sessions] == [old.id]
    assert SUMMARY_TEXT in context.past_conversations_text
    assert llm.calls == 1

    memory.load_context(new.id)  # digest is cached, no second LLM call
    assert llm.calls == 1


def test_digest_does_not_hide_old_session_messages(db):
    memory = make_memory(db)
    old = memory.start_session()
    memory.add_message(old.id, MessageRole.USER, "hello")
    memory.add_message(old.id, MessageRole.ASSISTANT, "hi")
    memory.load_context(memory.start_session().id)  # builds old session's digest

    reopened = memory.load_context(old.id, include_long_term=False)
    assert len(reopened.recent_messages) == 2
    assert reopened.session_summary == ""
