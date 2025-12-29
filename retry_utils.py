"""Retry utilities with exponential backoff for API calls.

Provides:
- exponential_backoff_delay: Calculate delay with jitter
- CircuitBreaker: Opens after consecutive failures, auto-recovers
- get_passio_circuit_breaker: Shared instance for Passio API
"""

import os
import random
import sys
import time
from typing import Optional


# Configuration (can be overridden via environment variables)
DEFAULT_MAX_RETRIES = int(os.getenv("PASSIO_MAX_RETRIES", "3"))
DEFAULT_BASE_DELAY_SECONDS = float(os.getenv("PASSIO_RETRY_BASE_DELAY", "1.0"))
DEFAULT_MAX_DELAY_SECONDS = float(os.getenv("PASSIO_RETRY_MAX_DELAY", "30.0"))
DEFAULT_EXPONENTIAL_BASE = 2.0
DEFAULT_JITTER_FACTOR = 0.1

# Errors that should trigger retry
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRIABLE_KEYWORDS = (
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "connection",
    "error while searching",
)


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and requests are blocked."""

    pass


class CircuitBreaker:
    """Simple circuit breaker implementation.

    Opens after `failure_threshold` consecutive failures.
    Half-opens after `recovery_timeout` seconds.
    Closes after first success in half-open state.

    States:
    - closed: Normal operation, requests go through
    - open: Requests blocked, waiting for recovery timeout
    - half-open: Testing if service recovered, one request allowed
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for logging
            failure_threshold: Open after this many consecutive failures
            recovery_timeout: Seconds before trying again after opening
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != "open":
                self.state = "open"
                print(
                    f"   ⚡ Circuit breaker '{self.name}' OPENED after {self.failure_count} failures",
                    file=sys.stderr,
                )

    def record_success(self) -> None:
        """Record a success and close the circuit if half-open."""
        if self.state == "half-open":
            print(
                f"   ⚡ Circuit breaker '{self.name}' CLOSED after success",
                file=sys.stderr,
            )
        self.failure_count = 0
        self.state = "closed"

    def can_execute(self) -> bool:
        """Check if requests can proceed.

        Returns:
            True if requests are allowed, False if blocked
        """
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if recovery timeout has elapsed
            elapsed = time.time() - (self.last_failure_time or 0)
            if elapsed >= self.recovery_timeout:
                self.state = "half-open"
                print(
                    f"   ⚡ Circuit breaker '{self.name}' HALF-OPEN, testing...",
                    file=sys.stderr,
                )
                return True
            return False

        # half-open: allow one request through
        return True

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


def exponential_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
) -> float:
    """Calculate delay with exponential backoff and jitter.

    Formula: min(base_delay * (exponential_base ** attempt), max_delay) + jitter

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential growth (default 2)
        jitter_factor: Random jitter as fraction of delay (default 0.1)

    Returns:
        Delay in seconds to wait before next attempt
    """
    delay = min(base_delay * (exponential_base**attempt), max_delay)
    jitter = delay * jitter_factor * random.uniform(-1, 1)
    return max(0, delay + jitter)


def is_retriable_error(exc: Exception) -> bool:
    """Check if an error is retriable (rate limit, server error, etc.).

    Args:
        exc: Exception to check

    Returns:
        True if the error should trigger a retry
    """
    # Check HTTP status code
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in RETRIABLE_STATUS_CODES:
        return True

    # Check error message
    try:
        error_msg = str(exc).lower()
    except Exception:
        error_msg = ""

    return any(kw in error_msg for kw in RETRIABLE_KEYWORDS)


# Global circuit breakers for shared APIs
_passio_circuit_breaker: Optional[CircuitBreaker] = None


def get_passio_circuit_breaker() -> CircuitBreaker:
    """Get the shared Passio API circuit breaker.

    Returns:
        CircuitBreaker instance for Passio API calls
    """
    global _passio_circuit_breaker
    if _passio_circuit_breaker is None:
        _passio_circuit_breaker = CircuitBreaker(
            name="passio_api",
            failure_threshold=int(os.getenv("PASSIO_CIRCUIT_FAILURE_THRESHOLD", "5")),
            recovery_timeout=float(
                os.getenv("PASSIO_CIRCUIT_RECOVERY_TIMEOUT", "60.0")
            ),
        )
    return _passio_circuit_breaker


def reset_passio_circuit_breaker() -> None:
    """Reset the Passio circuit breaker (useful for testing)."""
    global _passio_circuit_breaker
    if _passio_circuit_breaker is not None:
        _passio_circuit_breaker.reset()
