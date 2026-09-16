"""hypernix integration — model fetching, catalogue, health and web search.

The previous version of this file called `hypernix.fetch()` and
`hypernix.infer()`. Neither function exists, in 0.72.5 or any other
release, so every call raised `AttributeError` the moment it was reached
and the whole integration was decorative.

This is written against the real 0.72.5 surface:

  download_model / preheat    fetch and warm a model
  list_models / resolve_*     the model catalogue, so short names work
  healthcheck / diagnostic_info   what `nixorb check` reports
  search_web_non_api          web search with no API key to manage
  calculate_vram_context      how much context this card can actually hold

Everything degrades: hypernix is optional, and each call says what to
install rather than raising `AttributeError` three frames down.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The extra that installs it.
INSTALL_HINT = "pip install 'nixorb[hypernix]'"

#: The oldest release carrying the API this module calls.
MIN_VERSION = (0, 72, 5)


class HypernixUnavailable(RuntimeError):
    """hypernix is not installed, or is too old for what was asked."""


class HypernixClient:
    """Thin wrapper over the hypernix package, safe when it is absent."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._hn: Any = None
        self._version = ""
        try:
            import hypernix
        except ImportError:
            log.debug("hypernix not installed — %s", INSTALL_HINT)
            return

        self._hn = hypernix
        self._version = str(getattr(hypernix, "__version__", "") or "")
        log.info("hypernix %s available", self._version or "unknown")

    # ── availability ─────────────────────────────────────────────── #

    @property
    def version(self) -> str:
        return self._version

    def is_available(self) -> bool:
        return self._hn is not None

    def supports(self, name: str) -> bool:
        """Is this entry point present in the installed hypernix?"""
        return self._hn is not None and hasattr(self._hn, name)

    def _require(self, name: str | None = None) -> Any:
        if self._hn is None:
            raise HypernixUnavailable(
                f"hypernix is not installed. Install it with: {INSTALL_HINT}"
            )
        if name is not None and not hasattr(self._hn, name):
            want = ".".join(str(n) for n in MIN_VERSION)
            raise HypernixUnavailable(
                f"hypernix {self._version or 'installed'} has no '{name}'. "
                f"NixOrb needs {want} or newer: pip install -U hypernix"
            )
        return self._hn

    def _token(self) -> str | None:
        from nixorb import hf

        return hf.token(self._settings)

    # ── models ───────────────────────────────────────────────────── #

    async def download_model(
        self, repo_id: str, *, revision: str | None = None,
        local_dir: str | Path | None = None,
    ) -> Path:
        """Fetch a model and return where it landed."""
        hn = self._require("download_model")
        log.info("hypernix: downloading %s", repo_id)
        return await asyncio.to_thread(
            lambda: hn.download_model(
                repo_id,
                revision=revision,
                local_dir=str(local_dir) if local_dir else None,
                token=self._token(),
            )
        )

    async def preheat(self, repo_id: str, *, device: str | None = None) -> Any:
        """Download *and* load a model, returning hypernix's runner."""
        hn = self._require("preheat")
        log.info("hypernix: preheating %s", repo_id)
        return await asyncio.to_thread(
            lambda: hn.preheat(repo_id, device=device, token=self._token())
        )

    def resolve_repo_id(self, name: str) -> str:
        """Expand a short catalogue name into a full repo id.

        Lets `llm_model = "qwen3.5-4b"` mean what it looks like it means,
        rather than only accepting a full `owner/name`.
        """
        if not self.supports("resolve_repo_id"):
            return name
        try:
            return str(self._hn.resolve_repo_id(name) or name)
        except Exception as exc:
            log.debug("hypernix: could not resolve %r (%s)", name, exc)
            return name

    def list_models(
        self, *, contains: str | None = None, arch: str | None = None
    ) -> list[tuple[str, str, str]]:
        """The catalogue as (name, repo_id, notes) rows."""
        if not self.supports("list_models"):
            return []
        try:
            return list(self._hn.list_models(filter_substring=contains, arch=arch))
        except Exception as exc:
            log.debug("hypernix: list_models failed (%s)", exc)
            return []

    # ── diagnostics ──────────────────────────────────────────────── #

    def healthcheck(self) -> dict[str, Any]:
        """hypernix's own view of this machine, flattened to a dict."""
        if not self.supports("healthcheck"):
            return {}
        try:
            report = self._hn.healthcheck()
        except Exception as exc:
            log.debug("hypernix: healthcheck failed (%s)", exc)
            return {}

        fields = (
            "hypernix_version", "python_version", "platform", "torch_version",
            "cuda_available", "cuda_device_count", "cuda_device_names",
            "known_models_count",
        )
        return {name: getattr(report, name, None) for name in fields}

    def diagnostic_info(self) -> dict[str, Any]:
        if not self.supports("diagnostic_info"):
            return {}
        try:
            return dict(self._hn.diagnostic_info())
        except Exception as exc:
            log.debug("hypernix: diagnostic_info failed (%s)", exc)
            return {}

    # ── planning ─────────────────────────────────────────────────── #

    def context_for_vram(
        self, vram_gb: float, params_billions: float, *, precision: str = "fp16"
    ) -> int | None:
        """How many tokens of context this card can hold for that model."""
        if not self.supports("calculate_vram_context"):
            return None
        try:
            return int(
                self._hn.calculate_vram_context(
                    vram_gb=vram_gb,
                    model_size_params=params_billions,
                    batch_size=1,
                    precision=precision,
                )
            )
        except Exception as exc:
            log.debug("hypernix: calculate_vram_context failed (%s)", exc)
            return None

    # ── web search ───────────────────────────────────────────────── #

    async def search_web(
        self, query: str, *, max_results: int = 5, engine: str = "auto"
    ) -> list[dict[str, str]]:
        """Search the web with no API key to obtain or rotate."""
        hn = self._require("search_web_non_api")
        return await asyncio.to_thread(
            lambda: list(
                hn.search_web_non_api(query, max_results=max_results, engine=engine)
            )
        )
