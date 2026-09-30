"""
ASTRA INTEL — LLM Manager (Day 2)
====================================
Unified LLM interface with explicit Cloud vs Air-Gapped mode switching.
Supports Groq, Google Gemini, OpenAI (Cloud) and Ollama (Local/Air-Gapped).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# ── Enums & Config ──────────────────────────────────────────────────────────

class InferenceMode(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"


class CloudProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class ProviderStatus:
    """Status object rendered in the sidebar."""
    mode: str            # "CLOUD" or "LOCAL (AIR-GAPPED)"
    provider: str        # e.g. "GROQ", "OLLAMA"
    model: str
    status: str          # "Ready", "Unreachable", "No Key"
    api_key_preview: str # masked key or ""


# ── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified wrapper that routes calls to Cloud APIs or local Ollama.

    Parameters
    ----------
    mode : InferenceMode
        CLOUD uses the first available cloud API key; LOCAL uses Ollama.
    cloud_provider : CloudProvider or None
        Force a specific cloud provider.  None = auto-detect by key priority.
    ollama_model : str
        Model name for Ollama (e.g. "llama3.2", "mistral", "qwen2.5").
    """

    def __init__(
        self,
        mode: InferenceMode = InferenceMode.CLOUD,
        cloud_provider: Optional[CloudProvider] = None,
        ollama_model: Optional[str] = None,
    ):
        self.mode = mode
        self._cloud_provider = cloud_provider
        self._ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self._ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Resolve cloud provider + key eagerly so status is available
        self._provider: str = ""
        self._api_key: str = ""
        self._model: str = ""

        if self.mode == InferenceMode.CLOUD:
            self._resolve_cloud(cloud_provider)
        else:
            self._provider = "ollama"
            self._api_key = ""
            self._model = self._ollama_model

    # ── Resolution ──────────────────────────────────────────────────────────

    def _resolve_cloud(self, forced: Optional[CloudProvider] = None) -> None:
        """Pick the first available cloud provider based on env keys."""
        candidates = [
            (CloudProvider.GROQ, "GROQ_API_KEY", os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")),
            (CloudProvider.GEMINI, "GOOGLE_API_KEY", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
            (CloudProvider.OPENAI, "OPENAI_API_KEY", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        ]

        if forced:
            candidates = [c for c in candidates if c[0] == forced]

        for prov, env_var, default_model in candidates:
            key = os.getenv(env_var, "").strip()
            if key:
                self._provider = prov.value
                self._api_key = key
                self._model = default_model
                return

        # No cloud key found — mark as unavailable; caller should switch to LOCAL
        self._provider = "none"
        self._api_key = ""
        self._model = ""

    # ── Public API ──────────────────────────────────────────────────────────

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a system+user message pair to the active LLM and return the text.

        Raises RuntimeError on any API / connectivity failure.
        """
        try:
            if self.mode == InferenceMode.LOCAL:
                return self._call_ollama(system_prompt, user_prompt)

            if self._provider == "groq":
                return self._call_groq(system_prompt, user_prompt)
            elif self._provider == "gemini":
                return self._call_gemini(system_prompt, user_prompt)
            elif self._provider == "openai":
                return self._call_openai(system_prompt, user_prompt)
            else:
                raise RuntimeError(
                    "No cloud API key configured. "
                    "Please set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in .env, "
                    "or switch to Local/Air-Gapped mode."
                )
        except requests.exceptions.ConnectionError as exc:
            svc = "Ollama" if self.mode == InferenceMode.LOCAL else self._provider.upper()
            raise RuntimeError(
                f"Connection to {svc} failed. "
                f"{'Ensure `ollama serve` is running.' if self.mode == InferenceMode.LOCAL else 'Check your network.'}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("LLM request timed out. Please try again.") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM call failed ({self._provider}/{self._model}): {exc}") from exc

    def get_status(self) -> ProviderStatus:
        """Return a status snapshot for UI rendering."""
        masked = ""
        if self._api_key:
            masked = self._api_key[:6] + "..." + self._api_key[-4:] if len(self._api_key) > 10 else "***"

        if self.mode == InferenceMode.LOCAL:
            reachable = self._check_ollama_health()
            return ProviderStatus(
                mode="LOCAL (AIR-GAPPED)",
                provider="OLLAMA",
                model=self._ollama_model,
                status="Ready" if reachable else "Unreachable",
                api_key_preview="N/A",
            )

        if not self._api_key:
            return ProviderStatus(
                mode="CLOUD",
                provider=self._provider.upper() or "NONE",
                model=self._model or "—",
                status="No Key",
                api_key_preview="",
            )

        return ProviderStatus(
            mode="CLOUD",
            provider=self._provider.upper(),
            model=self._model,
            status="Ready",
            api_key_preview=masked,
        )

    # ── Provider Implementations ────────────────────────────────────────────

    def _call_groq(self, sys: str, usr: str) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, sys: str, usr: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": sys}]},
            "contents": [{"parts": [{"text": usr}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, sys: str, usr: str) -> str:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, sys: str, usr: str) -> str:
        resp = requests.post(
            f"{self._ollama_base}/api/chat",
            json={
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _check_ollama_health(self) -> bool:
        """Quick liveness check for Ollama server."""
        try:
            resp = requests.get(f"{self._ollama_base}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
