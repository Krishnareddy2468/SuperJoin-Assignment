"""Runtime configuration.

Resolved once, here, so that no other module needs to touch ``os.environ`` or
guess at filesystem layout. Everything downstream takes a :class:`Config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a convenience, not a requirement
    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ENV_FILE = PROJECT_ROOT / ".env"

# Default extraction model: Gemini Flash. Chosen for a usable free tier (so
# graders can run the LLM path without our account), a large context window,
# and native JSON-schema-constrained output.
#
# Model IDs move; don't trust this constant blindly. `python -m factlayer.cli
# models` lists what the configured key can actually reach.
DEFAULT_LLM_MODEL = "gemini-2.5-flash"

# Tried in order when the preferred model will not answer, so one unavailable model or
# an exhausted free-tier quota degrades the run instead of ending it. Cheaper, higher
# quota models sit later in the list on purpose. Confirm what a given key can actually
# reach with `python -m factlayer.cli models --check` rather than trusting these names.
DEFAULT_LLM_FALLBACK_MODELS = (
    # Cheaper and higher quota than the preferred model, so it is the natural next stop
    # when a free-tier limit is what went wrong.
    "gemini-2.5-flash-lite",
    # A moving alias rather than a pinned name, kept last on purpose: if both pinned
    # models are ever retired this still resolves to something current, at the cost of
    # not being reproducible. Verified against a live key, unlike a name recalled from
    # memory - "gemini-2.0-flash" sat here first and turned out not to exist for the key.
    "gemini-flash-latest",
)

# How many provider calls may be in flight at once. The calls are network-bound, so a
# handful of threads turns a document that took minutes into one that takes seconds.
# Kept modest on purpose: free-tier keys have per-minute limits, and hammering them
# trades a slow run for a failed one.
DEFAULT_LLM_MAX_WORKERS = 8

# The most provider calls one document may spend. A 27-page deck needs 9; a 100-page
# annual report would ask for 632. Without a ceiling a single large upload looks
# indistinguishable from a hang, so the LLM stops at the limit, says so, and the
# document still completes on deterministic facts.
DEFAULT_LLM_CALL_BUDGET = 150

# Seconds to wait for one model call, and how many attempts it gets.
#
# Both numbers come from measurement rather than taste. Across one document the honest
# calls took 2.4 to 20.7 seconds, so 25 leaves headroom for the slowest real answer. One
# passage instead burned the full timeout every time; at the original three attempts it
# held the document for 87 seconds while its neighbours had long finished.
#
# So: one attempt. A call that cannot answer inside 25 seconds is not going to answer on
# the second ask either, and the model fallback chain already covers the failure that
# retrying was there for. Fail that passage quickly, keep its deterministic facts, and
# let the rest of the document through.
DEFAULT_LLM_TIMEOUT_SECONDS = 25.0
DEFAULT_LLM_ATTEMPTS = 1

# Seconds the whole LLM phase may take for one document. Once it passes, passages still
# in flight are abandoned and reported; the deterministic facts are already safe. This
# is what stops one unlucky passage from deciding how long an upload takes.
DEFAULT_LLM_PHASE_DEADLINE = 60.0

# Bumped whenever the extraction prompt changes, so cached LLM responses from
# an older prompt are never silently reused.
PROMPT_VERSION = 1


def _env_models(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated model list, falling back to the built-in chain.

    Setting the variable to an empty string is a deliberate "no fallbacks", which is
    different from not setting it at all.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """Read a positive integer setting, ignoring anything that is not one."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0.1, float(raw.strip()))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Where things live, and whether the LLM extractor is available."""

    db_path: Path
    upload_dir: Path
    cache_dir: Path
    llm_model: str
    # Defaulted so that building a Config by hand - in a test, or a script that only
    # cares about paths - does not have to spell out the model chain.
    llm_fallback_models: tuple[str, ...] = DEFAULT_LLM_FALLBACK_MODELS
    llm_max_workers: int = DEFAULT_LLM_MAX_WORKERS
    llm_call_budget: int = DEFAULT_LLM_CALL_BUDGET
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    llm_attempts: int = DEFAULT_LLM_ATTEMPTS
    llm_phase_deadline: float = DEFAULT_LLM_PHASE_DEADLINE
    api_key: Optional[str] = None
    no_llm: bool = False

    @property
    def llm_models(self) -> tuple[str, ...]:
        """The preferred model first, then each fallback, with duplicates removed.

        Order is what the provider walks, so a model named as both the preference and a
        fallback must not be tried twice.
        """
        ordered = [self.llm_model, *self.llm_fallback_models]
        return tuple(dict.fromkeys(name for name in ordered if name))

    @property
    def llm_enabled(self) -> bool:
        """True only when an LLM call could actually succeed.

        Callers branch on this rather than on key presence, so that
        ``FACTLAYER_NO_LLM=1`` reliably forces the deterministic-only path.
        """
        return not self.no_llm and bool(self.api_key)

    def why_llm_disabled(self) -> Optional[str]:
        """Human-readable reason the LLM path is off, or None if it is on.

        Surfaced in CLI and API output: silently degrading to deterministic-only
        would misrepresent how a given set of results was produced.
        """
        if self.no_llm:
            return "FACTLAYER_NO_LLM is set"
        if not self.api_key:
            return "no GEMINI_API_KEY found in environment or .env"
        return None

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.upload_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config(
    *,
    db_path: Optional[Path] = None,
    no_llm: bool = False,
    data_dir: Optional[Path] = None,
    env_file: Optional[Path] = _PROJECT_ENV_FILE,
) -> Config:
    """Build a :class:`Config` from ``.env``, the environment, and overrides.

    Precedence: explicit argument > environment variable > built-in default.
    ``no_llm=True`` (the ``--no-llm`` flag) always wins over a present key.

    Pass ``env_file=None`` for a configuration that ignores the project's ``.env``.
    Tests need that: otherwise clearing the environment is undone by the file being
    read straight back in, and the suite would pass or fail depending on whether the
    developer running it happens to have a key on disk.
    """
    if env_file is not None:
        load_dotenv(env_file)

    base = data_dir or Path(os.environ.get("FACTLAYER_DATA_DIR", PROJECT_ROOT / "data"))

    return Config(
        db_path=db_path or base / "factlayer.db",
        upload_dir=base / "uploads",
        cache_dir=base / "cache",
        llm_model=os.environ.get("FACTLAYER_LLM_MODEL") or DEFAULT_LLM_MODEL,
        llm_fallback_models=_env_models(
            "FACTLAYER_LLM_FALLBACK_MODELS",
            DEFAULT_LLM_FALLBACK_MODELS,
        ),
        llm_max_workers=_env_int("FACTLAYER_LLM_MAX_WORKERS", DEFAULT_LLM_MAX_WORKERS, minimum=1),
        llm_call_budget=_env_int("FACTLAYER_LLM_CALL_BUDGET", DEFAULT_LLM_CALL_BUDGET, minimum=0),
        llm_timeout_seconds=_env_float(
            "FACTLAYER_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS
        ),
        llm_attempts=_env_int("FACTLAYER_LLM_ATTEMPTS", DEFAULT_LLM_ATTEMPTS, minimum=1),
        llm_phase_deadline=_env_float(
            "FACTLAYER_LLM_PHASE_DEADLINE", DEFAULT_LLM_PHASE_DEADLINE
        ),
        api_key=os.environ.get("GEMINI_API_KEY") or None,
        no_llm=no_llm or _env_flag("FACTLAYER_NO_LLM"),
    )
