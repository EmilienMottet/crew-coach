"""Observability utilities for structured logging and telemetry.

This module provides:
- Structured JSON logging to /tmp/crew_logs/
- Log rotation with configurable retention
- Context managers for workflow tracking
- Helper functions for logging complex data structures
"""

import json
import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# Configuration
# ============================================================================

LOG_DIR = Path(os.getenv("LOG_DIR", "/tmp/crew_logs"))
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "7"))
STRUCTURED_LOG_LEVEL = os.getenv("STRUCTURED_LOG_LEVEL", "INFO").upper()

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# JSON Formatter
# ============================================================================


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from extra parameter
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


# ============================================================================
# Logger Setup
# ============================================================================


def setup_structured_logger(name: str) -> logging.Logger:
    """Set up a structured logger that writes JSON to /tmp/crew_logs/.

    Args:
        name: Logger name (e.g., "crew.music", "crew.lyrics")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, STRUCTURED_LOG_LEVEL))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create rotating file handler (daily rotation)
    log_file = LOG_DIR / f"{name}.jsonl"
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


# ============================================================================
# Cleanup Old Logs
# ============================================================================


def cleanup_old_logs() -> None:
    """Remove log files older than LOG_RETENTION_DAYS."""
    try:
        cutoff_date = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)

        for log_file in LOG_DIR.glob("*.jsonl*"):
            # Check file modification time
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                print(f"🗑️  Deleted old log: {log_file.name}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Failed to cleanup old logs: {e}", file=sys.stderr)


# ============================================================================
# Context Managers
# ============================================================================


@contextmanager
def log_workflow(logger: logging.Logger, workflow_name: str, **context):
    """Context manager for tracking workflow execution.

    Example:
        with log_workflow(logger, "music_analysis", activity_id=12345):
            # ... workflow code ...
            pass
    """
    start_time = datetime.utcnow()

    logger.info(
        f"Workflow started: {workflow_name}",
        extra={
            "extra_fields": {
                "workflow": workflow_name,
                "phase": "start",
                **context,
            }
        },
    )

    try:
        yield
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Workflow completed: {workflow_name}",
            extra={
                "extra_fields": {
                    "workflow": workflow_name,
                    "phase": "complete",
                    "duration_ms": duration_ms,
                    **context,
                }
            },
        )
    except Exception as e:
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(
            f"Workflow failed: {workflow_name}",
            extra={
                "extra_fields": {
                    "workflow": workflow_name,
                    "phase": "error",
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    **context,
                }
            },
            exc_info=True,
        )
        raise


# ============================================================================
# Helper Functions
# ============================================================================


def log_data_structure(
    logger: logging.Logger,
    name: str,
    data: Any,
    level: str = "DEBUG",
) -> None:
    """Log a complex data structure with truncation for large payloads.

    Args:
        logger: Logger instance
        name: Description of the data
        data: Data to log (dict, list, etc.)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    log_method = getattr(logger, level.lower())

    # Serialize data
    try:
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data, indent=2)
        else:
            serialized = str(data)
    except Exception as e:
        serialized = f"<non-serializable: {type(data).__name__}>"
        logger.warning(f"Failed to serialize {name}: {e}")

    # Truncate if too large (> 5000 chars)
    if len(serialized) > 5000:
        truncated = (
            serialized[:5000] + f"\n... (truncated {len(serialized) - 5000} chars)"
        )
        log_method(
            f"{name} (truncated)",
            extra={
                "extra_fields": {
                    "data_name": name,
                    "data_preview": truncated,
                    "full_size": len(serialized),
                    "truncated": True,
                }
            },
        )
    else:
        log_method(
            name,
            extra={
                "extra_fields": {
                    "data_name": name,
                    "data": serialized,
                    "truncated": False,
                }
            },
        )


# ============================================================================
# Initialization
# ============================================================================

# Cleanup old logs on module import
cleanup_old_logs()

print(
    f"📊 Structured logging initialized:\n"
    f"   Log directory: {LOG_DIR}\n"
    f"   Retention: {LOG_RETENTION_DAYS} days\n"
    f"   Level: {STRUCTURED_LOG_LEVEL}\n",
    file=sys.stderr,
)
