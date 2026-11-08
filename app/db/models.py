"""ORM models (database tables) and Pydantic schemas (API contracts).

Why two kinds of classes live here
----------------------------------
* **ORM models** describe how data is *stored* (tables, constraints, indexes).
* **Pydantic schemas** describe how data *enters and leaves* the API
  (validation, serialization).

Keeping them separate means we can change storage details without breaking
the API, and we never accidentally expose an internal column to clients.
(When the project grows, the schemas section can move to ``app/schemas/``
without any change in behaviour.)
"""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.core.config import get_settings
from app.db.database import Base

DEFAULT_SESSION_TITLE = "New conversation"
MAX_TITLE_LENGTH = 200


# =============================================================================
# Helpers
# =============================================================================
def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    ``datetime.utcnow()`` returns a *naive* value (deprecated in 3.12) which is
    easy to mix up with local time. Always-aware UTC avoids that whole class of bug.
    """
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Random UUID4 string, used as public primary key for chat sessions.

    UUIDs (vs. 1, 2, 3...) are not guessable, so one user cannot enumerate
    another user's conversations by incrementing an ID in the URL.
    """
    return str(uuid.uuid4())


class UTCDateTime(TypeDecorator):
    """DateTime column that always stores UTC and always returns aware values.

    SQLite has no real timezone support: it silently drops tzinfo, so values
    come back naive. Without this decorator, comparing a stored timestamp with
    ``utcnow()`` raises "can't compare offset-naive and offset-aware datetimes".
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            # Refuse naive datetimes instead of guessing their timezone.
            raise ValueError("Naive datetime is not allowed; use timezone-aware UTC values.")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            # SQLite returned a naive value; we know we stored UTC.
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class MessageRole(str, enum.Enum):
    """Who authored a message. Mirrors the roles used by chat LLM APIs.

    Inheriting from ``str`` makes members JSON-serialisable and directly
    comparable with plain strings.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# =============================================================================
# ORM models
# =============================================================================
class ChatSession(Base):
    """One conversation between a user and the advisor."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(
        String(MAX_TITLE_LENGTH), nullable=False, default=DEFAULT_SESSION_TITLE
    )
    # Soft-archive instead of delete keeps history recoverable and cheap to hide in the UI.
    is_archived: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    # cascade="all, delete-orphan": deleting a session removes its messages in
    # the ORM too. passive_deletes=True lets the DB-level ON DELETE CASCADE do
    # the work efficiently instead of loading every message first.
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",  # insertion order == conversation order
    )

    # The sidebar lists sessions newest-activity-first.
    __table_args__ = (Index("ix_chat_sessions_updated_at", "updated_at"),)

    def touch(self) -> None:
        """Bump ``updated_at`` (call when a message is added).

        ``onupdate`` only fires when the session row itself changes, so adding
        a message would not otherwise move the conversation to the top of the list.
        """
        self.updated_at = utcnow()

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id!r} title={self.title!r}>"


class Message(Base):
    """A single message inside a chat session."""

    __tablename__ = "messages"

    # Integer PK: messages are internal rows, and monotonically increasing IDs
    # give us a reliable ordering even if two messages share a timestamp.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(
        # native_enum=False -> stored as VARCHAR (SQLite has no ENUM type) and
        # values_callable stores "user" rather than the member name "USER".
        Enum(
            MessageRole,
            native_enum=False,
            length=16,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Reserved for Phase 2: which document chunks backed an answer (RAG
    # citations). Stored as JSON so the shape can evolve without a migration.
    sources: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    # Reserved for Phase 2: lets us monitor local-model usage / latency.
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    # lazy="selectin": loads feedback for many messages in ONE extra query,
    # avoiding the N+1 problem when serialising a whole conversation.
    feedback: Mapped[Optional["Feedback"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
        lazy="selectin",
    )

    # Loading a conversation filters by session and orders by time/id.
    __table_args__ = (Index("ix_messages_session_id_id", "session_id", "id"),)

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role.value!r} session={self.session_id!r}>"


class Feedback(Base):
    """User rating (thumbs up/down) for one assistant message.

    Collected so answer quality can be reviewed and prompts/models improved later.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    # +1 = helpful, -1 = not helpful. Integer (not bool) leaves room for a
    # richer scale later while the CHECK keeps today's data clean.
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    message: Mapped["Message"] = relationship(back_populates="feedback")

    __table_args__ = (
        # One feedback per message; re-rating should update, not duplicate.
        UniqueConstraint("message_id", name="uq_feedback_message_id"),
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating_valid"),
    )

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} message_id={self.message_id} rating={self.rating}>"


# =============================================================================
# Pydantic schemas (API validation / serialization)
# =============================================================================
class _ORMSchema(BaseModel):
    """Base for response schemas: allows building them from ORM objects.

    ``from_attributes=True`` lets ``MessageRead.model_validate(orm_message)``
    work, so routes can return ORM rows directly.
    """

    model_config = ConfigDict(from_attributes=True)


# --- Feedback ----------------------------------------------------------------
class FeedbackCreate(BaseModel):
    """Payload for rating an assistant message."""

    rating: Literal[-1, 1] = Field(description="1 = helpful, -1 = not helpful")
    comment: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("comment")
    @classmethod
    def _blank_comment_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Store ``None`` instead of an empty/whitespace-only comment."""
        if value is None:
            return None
        value = value.strip()
        return value or None


class FeedbackRead(_ORMSchema):
    id: int
    message_id: int
    rating: int
    comment: Optional[str] = None
    created_at: datetime


# --- Messages ----------------------------------------------------------------
class MessageCreate(BaseModel):
    """Payload for a new message.

    Clients may only create *user* messages through the API; assistant and
    system messages are produced server-side, so a client cannot forge what
    the "assistant" said.
    """

    content: str = Field(description="Message text.")

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content must not be empty.")
        # Read the limit at validation time (not import time) so it follows
        # the current settings, including test overrides.
        max_length = get_settings().max_message_length
        if len(value) > max_length:
            raise ValueError(f"Message is too long ({len(value)} > {max_length} characters).")
        return value


class MessageRead(_ORMSchema):
    id: int
    session_id: str
    role: MessageRole
    content: str
    sources: Optional[list[dict[str, Any]]] = None
    model_name: Optional[str] = None
    created_at: datetime
    feedback: Optional[FeedbackRead] = None


# --- Chat sessions -----------------------------------------------------------
class _TitleMixin(BaseModel):
    """Shared title cleaning for create/update payloads."""

    @field_validator("title", check_fields=False)
    @classmethod
    def _clean_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # Collapse newlines/tabs/multiple spaces so titles stay one tidy line in the sidebar.
        value = " ".join(value.split())
        if not value:
            raise ValueError("Title must not be blank.")
        if len(value) > MAX_TITLE_LENGTH:
            raise ValueError(f"Title must be at most {MAX_TITLE_LENGTH} characters.")
        return value


class ChatSessionCreate(_TitleMixin):
    """Payload for starting a conversation; title is optional."""

    title: Optional[str] = None


class ChatSessionUpdate(_TitleMixin):
    """Payload for renaming or (un)archiving; only provided fields change (PATCH)."""

    title: Optional[str] = None
    is_archived: Optional[bool] = None


class ChatSessionRead(_ORMSchema):
    """Session summary for the sidebar list (no messages, so it stays light)."""

    id: str
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class ChatSessionDetail(ChatSessionRead):
    """Full conversation including messages in chronological order."""

    messages: List[MessageRead] = Field(default_factory=list)
