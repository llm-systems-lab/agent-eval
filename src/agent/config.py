"""Environment-backed configuration for the agent and the evaluation judge.

Agent and judge settings are deliberately independent: building the agent
requires only the Google credentials, and building the judge requires only the
OpenRouter credentials. Secrets are stored as ``SecretStr`` so they are masked
in logs and repr output.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_AGENT_TEMPERATURE = 0.0
DEFAULT_LANGSMITH_PROJECT = "agenteval"


def _load_environment(env_path: str | Path | None = None) -> None:
    """Load a ``.env`` file without overriding already-set variables."""
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    load_dotenv(dotenv_path=path, override=False)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value.strip()


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a number.") from exc


class AgentSettings(BaseModel):
    """Google Gemini agent configuration."""

    google_api_key: SecretStr
    model: str = Field(min_length=1)
    temperature: float = DEFAULT_AGENT_TEMPERATURE


class JudgeSettings(BaseModel):
    """Evaluation judge configuration (Groq model via OpenRouter)."""

    openrouter_api_key: SecretStr
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)


class LangSmithSettings(BaseModel):
    """LangSmith tracing configuration."""

    tracing: bool = False
    api_key: SecretStr | None = None
    project: str = DEFAULT_LANGSMITH_PROJECT

    def apply(self) -> None:
        """Publish tracing settings to the environment LangSmith reads."""
        os.environ["LANGSMITH_TRACING"] = "true" if self.tracing else "false"
        os.environ["LANGSMITH_PROJECT"] = self.project
        if self.api_key is not None:
            os.environ["LANGSMITH_API_KEY"] = self.api_key.get_secret_value()


def load_agent_settings(env_path: str | Path | None = None) -> AgentSettings:
    _load_environment(env_path)
    return AgentSettings(
        google_api_key=SecretStr(_required_env("GOOGLE_API_KEY")),
        model=_required_env("AGENT_MODEL"),
        temperature=_float_env("AGENT_TEMPERATURE", default=DEFAULT_AGENT_TEMPERATURE),
    )


def load_judge_settings(env_path: str | Path | None = None) -> JudgeSettings:
    _load_environment(env_path)
    return JudgeSettings(
        openrouter_api_key=SecretStr(_required_env("OPENROUTER_API_KEY")),
        base_url=os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
        model=_required_env("JUDGE_MODEL"),
    )


def load_langsmith_settings(env_path: str | Path | None = None) -> LangSmithSettings:
    _load_environment(env_path)
    api_key = os.environ.get("LANGSMITH_API_KEY")
    return LangSmithSettings(
        tracing=_bool_env("LANGSMITH_TRACING", default=False),
        api_key=SecretStr(api_key) if api_key and api_key.strip() else None,
        project=os.environ.get("LANGSMITH_PROJECT") or DEFAULT_LANGSMITH_PROJECT,
    )
