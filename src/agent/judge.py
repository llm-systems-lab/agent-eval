"""OpenRouter-backed judge model for DeepEval metrics.

DeepEval graders call this model. It is an OpenAI-compatible client pointed at
OpenRouter, using the existing ``JudgeSettings`` (``OPENROUTER_API_KEY``,
``OPENROUTER_BASE_URL``, ``JUDGE_MODEL``). The agent's Gemini model is not used
for judging.
"""

from __future__ import annotations

from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import JudgeSettings, load_judge_settings


class OpenRouterJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """DeepEval judge backed by a Groq model hosted on OpenRouter."""

    def __init__(self, settings: JudgeSettings) -> None:
        self._settings = settings
        self._model = ChatOpenAI(
            model=settings.model,
            api_key=settings.openrouter_api_key,
            base_url=settings.base_url,
            temperature=0,
        )

    def load_model(self) -> Any:
        return self._model

    def get_model_name(self) -> str:
        return self._settings.model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if schema is not None:
            return self._model.with_structured_output(schema).invoke(prompt)
        return self._model.invoke(prompt).content

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        if schema is not None:
            return await self._model.with_structured_output(schema).ainvoke(prompt)
        return (await self._model.ainvoke(prompt)).content


def create_judge_model(settings: JudgeSettings | None = None) -> OpenRouterJudge:
    """Build the judge model from the environment or injected settings."""
    return OpenRouterJudge(settings or load_judge_settings())
