"""SQLAlchemy engine, session factory and helpers for SQLite.

Why this module exists
----------------------
All database plumbing is isolated here so the rest of the application only
deals with ``Session`` objects. Swapping the database later (e.g. PostgreSQL
for a multi-user deployment) should mean changing this file and the config,
not touching business logic.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class that every ORM model inherits from.

    Keeping it in this module (and models in ``models.py``) avoids circular
    imports: ``models`` imports ``database``, never the other way around.
    """


class DatabaseError(RuntimeError):
    """Raised for database problems that the caller can reasonably handle.

    Wrapping low-level SQLAlchemy errors lets API routes map them to a clean
    HTTP 500/503 response without importing SQLAlchemy themselves.
    """


def _configure_sqlite_connection(dbapi_connection: Any, _connection_record: Any) -> None:
    """Apply per-connection SQLite PRAGMAs.

    * ``foreign_keys=ON``: SQLite ignores foreign keys by default! Without this,
      deleting a chat session would NOT cascade to its messages and we would
      accumulate orphan rows.
    * ``journal_mode=WAL``: lets readers (Streamlit polling history) and a
      writer (saving a streamed answer) work concurrently instead of
      blocking each other with "database is locked" errors.
    * ``busy_timeout``: wait up to 5 s for a lock rather than failing instantly.
    * ``synchronous=NORMAL``: safe with WAL and noticeably faster writes.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def create_db_engine(database_url: str, echo: bool = False) -> Engine:
    """Build an Engine for ``database_url``.

    Exposed as a factory (instead of only a module-level engine) so tests can
    create an isolated in-memory engine without touching real data.

    Raises:
        DatabaseError: If the URL is invalid or the DB directory is unusable.
    """
    try:
        url = make_url(database_url)
    except Exception as exc:  # make_url raises ArgumentError; keep broad for clarity
        raise DatabaseError(f"Invalid database URL: {database_url!r}") from exc

    is_memory = url.database in (None, "", ":memory:")
    engine_kwargs: dict[str, Any] = {"echo": echo, "future": True}

    if is_memory:
        # An in-memory SQLite DB exists per connection. StaticPool forces all
        # sessions to share ONE connection, otherwise each session would see
        # an empty database and tables "disappear".
        engine_kwargs["poolclass"] = StaticPool
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Make sure the folder exists so SQLite doesn't fail with the cryptic
        # "unable to open database file".
        try:
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(
                f"Cannot create database directory for '{url.database}': {exc}"
            ) from exc
        # FastAPI runs sync endpoints in a thread pool, so one connection may
        # be used by a different thread than the one that created it.
        # SQLAlchemy's pool guarantees a connection is only used by one thread
        # at a time, so disabling SQLite's thread check is safe here.
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(url, **engine_kwargs)
    event.listen(engine, "connect", _configure_sqlite_connection)
    return engine


_settings = get_settings()
engine: Engine = create_db_engine(_settings.database_url, echo=_settings.database_echo)

# expire_on_commit=False: after commit() we still return ORM objects to the API
# layer for serialization. With the default (True) every attribute access would
# trigger a new SELECT, or raise DetachedInstanceError once the session closes.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields one session per request.

    The session is *always* closed (returning the connection to the pool), and
    rolled back if the request failed, so a half-finished transaction can never
    leak into the next request.

    Commits are intentionally left to the service layer, which knows what a
    complete unit of work is.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope(factory: Optional[sessionmaker] = None) -> Generator[Session, None, None]:
    """Transactional scope for code running outside FastAPI (scripts, tests, Streamlit).

    Commits on success, rolls back on any error, always closes. Low-level
    SQLAlchemy errors are re-raised as ``DatabaseError`` with the original
    exception chained, so callers get a stable exception type.
    """
    session = (factory or SessionLocal)()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Database transaction failed and was rolled back.")
        raise DatabaseError("Database operation failed.") from exc
    except Exception:
        # Non-database errors (e.g. validation) still must roll back.
        session.rollback()
        raise
    finally:
        session.close()


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all tables that do not yet exist (idempotent).

    Models are imported *inside* the function so they register themselves on
    ``Base.metadata`` before ``create_all`` runs, without creating a circular
    import at module load time.

    For schema changes after the first release, introduce Alembic migrations;
    ``create_all`` never alters existing tables.

    Raises:
        DatabaseError: If table creation fails.
    """
    from app.db import models  # noqa: F401  (import for side effect: registers tables)

    bind = target_engine or engine
    try:
        Base.metadata.create_all(bind=bind)
        logger.info("Database initialised (%s).", bind.url.render_as_string(hide_password=True))
    except SQLAlchemyError as exc:
        logger.exception("Failed to initialise the database.")
        raise DatabaseError("Could not initialise the database.") from exc


def check_db_connection(target_engine: Optional[Engine] = None) -> bool:
    """Return True if a trivial query succeeds. Used by the /health endpoint.

    Never raises: a health check must report failure, not crash.
    """
    try:
        with (target_engine or engine).connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        logger.exception("Database health check failed.")
        return False
