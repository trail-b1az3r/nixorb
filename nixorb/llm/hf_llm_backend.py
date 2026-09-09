"""nixorb/llm/hf_llm_backend.py — any local HuggingFace LLM, GGUF or full-precision.

Mirrors ``OllamaBackend``'s interface exactly (``stream``, ``last_tool_calls``,
``health_check``, ``close``, ``.model``, ``generate``) so main.py's
tool-calling loop and turn handling work unmodified regardless of which
backend ``build_llm`` returned.

Two model kinds, auto-detected from ``settings.llm_model`` /
``settings.llm_gguf_file``:

  * GGUF, via llama-cpp-python — used when ``llm_gguf_file`` is set, or the
    model string names a ``.gguf`` file, or the repo id looks like a GGUF
    repo (e.g. anything under ``*-GGUF``). Much lighter on the GTX 1080's
    8GB than full precision.
  * Full-precision / safetensors, via transformers — everything else. Loads
    with ``device_map="auto"`` and fp16 on CUDA, matching this project's
    documented Pascal constraints (no bf16, no flash-attn2).

Tool calling has no standard across arbitrary HF checkpoints the way it does
for Ollama's API, so this implements the convention several popular
instruct models (Qwen2.5, Hermes, Llama-3.1 fine-tunes) already speak
natively via their chat template: a ``<tool_call>{"name": ..., "arguments":
{...}}</tool_call>`` tag in the output. For models whose chat template
doesn't accept a ``tools=`` argument, the same convention is requested
explicitly via an appended system instruction, so tool calling degrades
gracefully instead of silently doing nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_SENTINEL = object()
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_TOOL_FALLBACK_INSTRUCTIONS = (
    "\n\nYou have access to tools. If you need to call one, respond with "
    "exactly one line of the form:\n"
    "<tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>\n"
    "Available tools:\n{tool_list}"
)


class HuggingFaceLLMError(RuntimeError):
    """Raised when a local HF model can't be loaded or fails to generate."""


def _looks_like_gguf(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.endswith(".gguf") or "gguf" in Path(lowered).name


class HuggingFaceLLMBackend:
    """Local LLM backend for any transformers or GGUF checkpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_id = settings.llm_model
        self._gguf_file = getattr(settings, "llm_gguf_file", "") or ""
        self._token = getattr(settings, "hf_token", "") or None
        self._use_gguf = bool(self._gguf_file) or _looks_like_gguf(self._model_id)

        self._llama: Any = None  # llama_cpp.Llama instance
        self._hf_model: Any = None  # transformers model
        self._tokenizer: Any = None
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model_id

    async def close(self) -> None:
        """No persistent connection to close — kept for interface parity."""
        return None

    # ── loading ──────────────────────────────────────────────────── #

    def _load_gguf(self) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            from nixorb.hf import explain_import_error

            raise HuggingFaceLLMError(
                explain_import_error(exc, "llama_cpp", "GGUF models")
            ) from exc

        n_gpu_layers = -1  # offload everything; llama.cpp falls back on OOM
        try:
            if self._gguf_file:
                return Llama.from_pretrained(
                    repo_id=self._model_id,
                    filename=self._gguf_file,
                    n_ctx=4096,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            if Path(self._model_id).is_file():
                return Llama(
                    model_path=self._model_id,
                    n_ctx=4096,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            # A bare repo id with no filename: let llama-cpp-python pick a
            # default quant out of the repo (its own "*Q4_K_M*" heuristic).
            return Llama.from_pretrained(
                repo_id=self._model_id,
                filename="*.gguf",
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise HuggingFaceLLMError(
                f"Could not load GGUF model '{self._model_id}'"
                f"{' / ' + self._gguf_file if self._gguf_file else ''}: "
                f"{self._describe_gguf_failure(exc)}"
            ) from exc

    @staticmethod
    def _describe_gguf_failure(exc: Exception) -> str:
        """llama.cpp says "Failed to load model from file" and nothing else.

        That one line covers a truncated download, a file that is not a
        GGUF at all, and a good GGUF whose architecture the bundled
        llama.cpp is too old to build — three problems with three
        different fixes. The file's own header says which.
        """
        import re

        match = re.search(r"Failed to load model from file:\s*(\S.*)", str(exc))
        if not match:
            return str(exc)

        from nixorb.llm.gguf_probe import explain_load_failure

        return explain_load_failure(match.group(1).strip(), exc)

    def _load_transformers(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            # torch is imported here too, and it is deliberately not one of
            # this project's dependencies — so this branch blamed
            # transformers for a missing torch, and pointed at an extra
            # that would not have installed torch even if it existed.
            from nixorb.hf import explain_import_error

            raise HuggingFaceLLMError(
                explain_import_error(exc, "transformers", "The HuggingFace LLM backend")
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_id, token=self._token)
            # GTX 1080 is Pascal: fp16 works, bf16 does not.
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model: Any = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                token=self._token,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                model = model.to("cpu")
            return model, tokenizer
        except Exception as exc:
            raise HuggingFaceLLMError(
                f"Could not load HuggingFace model '{self._model_id}': {exc}"
            ) from exc

    def _ensure_loaded(self) -> None:
        if self._use_gguf:
            if self._llama is None:
                log.info("LLM: loading GGUF model '%s'", self._model_id)
                self._llama = self._load_gguf()
                log.info("LLM: GGUF model loaded")
        else:
            if self._hf_model is None:
                log.info("LLM: loading HuggingFace model '%s'", self._model_id)
                self._hf_model, self._tokenizer = self._load_transformers()
                log.info("LLM: HuggingFace model loaded")

    # ── health ───────────────────────────────────────────────────── #

    async def health_check(self) -> dict[str, Any]:
        """Best-effort load check. Never raises — loading a multi-GB model
        just to answer "is this configured right" would defeat the point of
        a health check, so this only validates that the repo/path exists
        and the required library is importable, not that the full model
        loads cleanly (that happens lazily on first use).
        """
        try:
            if self._use_gguf:
                try:
                    import llama_cpp  # noqa: F401
                except ImportError:
                    return {
                        "ok": False,
                        "error": "llama-cpp-python is not installed. Install "
                                 "with: pip install 'nixorb[llama_cpp]'",
                        "models": [],
                    }
            else:
                try:
                    import transformers  # noqa: F401
                except ImportError as exc:
                    from nixorb.hf import explain_import_error

                    return {
                        "ok": False,
                        "error": explain_import_error(
                            exc, "transformers", "The HuggingFace LLM backend"
                        ),
                        "models": [],
                    }
                if not Path(self._model_id).exists():
                    from huggingface_hub import HfApi
                    from huggingface_hub.utils import HfHubHTTPError
                    try:
                        HfApi().model_info(self._model_id, token=self._token)
                    except HfHubHTTPError as exc:
                        return {
                            "ok": False,
                            "error": f"HuggingFace model '{self._model_id}' "
                                     f"not found or not accessible: {exc}",
                            "models": [],
                        }
                    except Exception:
                        # Offline / no network — don't block startup on this.
                        pass
            return {"ok": True, "error": "", "models": [self._model_id]}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "models": []}

    # ── prompt / tool-call plumbing ─────────────────────────────────── #

    def _build_prompt_messages(
        self, messages: list[dict], tools: list[dict] | None
    ) -> list[dict]:
        if not tools:
            return messages
        tool_list = "\n".join(
            f"- {t.get('function', t).get('name', '?')}: "
            f"{t.get('function', t).get('description', '')}"
            for t in tools
        )
        extra = _TOOL_FALLBACK_INSTRUCTIONS.format(tool_list=tool_list)
        out = list(messages)
        if out and out[0].get("role") == "system":
            out[0] = {**out[0], "content": out[0]["content"] + extra}
        else:
            out.insert(0, {"role": "system", "content": extra.strip()})
        return out

    def _extract_tool_calls(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Strip <tool_call> tags out of the visible text, parse them out."""
        calls: list[dict[str, Any]] = []

        def _pull(match: re.Match) -> str:
            try:
                obj = json.loads(match.group(1))
                if obj.get("name"):
                    calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
            except json.JSONDecodeError:
                log.debug("LLM: malformed tool_call block ignored: %r", match.group(1)[:200])
            return ""

        # No .strip() here. This runs on every streamed token, and llama.cpp
        # emits them with their leading space attached (" the", " quick").
        # Stripping each one ran the whole answer together —
        # "Iamthinkingaboutthis" — and dropped space-only tokens entirely,
        # because "" is falsy at the call site.
        cleaned = _TOOL_CALL_RE.sub(_pull, text)
        return cleaned, calls

    # ── generation ───────────────────────────────────────────────── #

    def _gguf_token_worker(
        self, prompt_messages: list[dict], out_q: queue.Queue
    ) -> None:
        try:
            for chunk in self._llama.create_chat_completion(
                messages=prompt_messages,
                max_tokens=int(self._settings.llm_max_tokens),
                temperature=float(self._settings.llm_temperature),
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    out_q.put(piece)
        except Exception as exc:
            out_q.put(exc)
        finally:
            out_q.put(_SENTINEL)

    def _transformers_token_worker(
        self, prompt_messages: list[dict], out_q: queue.Queue
    ) -> None:
        try:
            from transformers import TextIteratorStreamer

            try:
                prompt = self._tokenizer.apply_chat_template(
                    prompt_messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in prompt_messages
                ) + "\nassistant:"

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(self._settings.llm_max_tokens),
                temperature=max(float(self._settings.llm_temperature), 0.01),
                do_sample=float(self._settings.llm_temperature) > 0,
            )
            gen_thread = threading.Thread(
                target=self._hf_model.generate, kwargs=gen_kwargs, daemon=True
            )
            gen_thread.start()
            for piece in streamer:
                if piece:
                    out_q.put(piece)
            gen_thread.join()
        except Exception as exc:
            out_q.put(exc)
        finally:
            out_q.put(_SENTINEL)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        self.last_tool_calls = []

        await loop.run_in_executor(None, self._ensure_loaded)

        await bus.emit(
            Event.LLM_START, data={"model": self._model_id},
            source="HuggingFaceLLMBackend", priority=3,
        )

        prompt_messages = self._build_prompt_messages(messages, tools)
        out_q: queue.Queue = queue.Queue()
        worker = self._gguf_token_worker if self._use_gguf else self._transformers_token_worker
        threading.Thread(target=worker, args=(prompt_messages, out_q), daemon=True).start()

        raw_chunks: list[str] = []
        # Buffer text until a <tool_call> block (if any) is complete, rather
        # than streaming raw JSON tags to the UI/TTS mid-generation.
        pending = ""
        try:
            while True:
                item = await loop.run_in_executor(None, out_q.get)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise HuggingFaceLLMError(str(item)) from item

                raw_chunks.append(item)
                pending += item
                if "<tool_call>" in pending and "</tool_call>" not in pending:
                    # Hold back until the tag closes so we don't leak it.
                    continue
                cleaned, calls = self._extract_tool_calls(pending)
                self.last_tool_calls.extend(calls)
                pending = ""
                if cleaned:
                    await bus.emit(
                        Event.LLM_CHUNK, data={"chunk": cleaned},
                        source="HuggingFaceLLMBackend", priority=3,
                    )
                    yield cleaned
        except HuggingFaceLLMError:
            raise
        finally:
            if pending:
                cleaned, calls = self._extract_tool_calls(pending)
                self.last_tool_calls.extend(calls)
                if cleaned:
                    yield cleaned

        await bus.emit(
            Event.LLM_DONE, data={"model": self._model_id},
            source="HuggingFaceLLMBackend", priority=3,
        )

    async def generate(self, messages: list[dict]) -> str:
        chunks: list[str] = []
        async for chunk in self.stream(messages):
            chunks.append(chunk)
        return "".join(chunks)
