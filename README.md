# NixOrb 🌐

**Floating AI assistant orb for Arch Linux — KDE Plasma 6 Wayland — runs entirely on your machine**

A complete ground-up remake of NixOrb with a focus on local AI, rock-solid Qt6 stability, and clean modular architecture.

---

## Models

Every stage is pluggable. Point it at Ollama, at any model on the Hugging Face
Hub, or at NVIDIA Nemotron for streaming speech.

| Stage | Backends | Setting |
|-------|----------|---------|
| **Speech → text** | `faster-whisper` (default) · `huggingface` (any ASR model) · `nemotron` | `asr_backend` |
| **Thinking** | `ollama` (default) · `huggingface` (any causal LM) | `llm_backend` |
| **Text → speech** | `piper` (default) · `huggingface` (any TTS model) · `espeak` | `tts_backend` |

```bash
pip install 'nixorb[hf]'        # any Hugging Face ASR / TTS / LLM
pip install 'nixorb[nemotron]'  # NVIDIA Nemotron 3.5 streaming ASR
pip install 'nixorb[quant]'     # 4-bit LLMs (bitsandbytes)
```

> **Install NixOrb into its own virtualenv**, not a shared interpreter.
> The default ASR checkpoint needs `transformers >= 5.13`, and much of the
> local-AI ecosystem still pins `transformers <= 4.5x` — LLaMA-Factory,
> for one. Whichever you install second wins, pip prints a single
> `dependency conflicts` warning, and the loser breaks later from inside a
> model loader. `./install.sh` already builds one at
> `~/.local/share/nixorb/venv`; `nixorb check` tells you if you are in a
> shared environment and names anything you collide with.

### Any Hugging Face model

```toml
# Speech recognition — anything with an automatic-speech-recognition tag
asr_backend = "huggingface"
asr_model   = "openai/whisper-small"      # distil-whisper, MMS, Wav2Vec2, …

# Thinking — any causal LM, streamed token by token
llm_backend  = "huggingface"
llm_hf_model = "Qwen/Qwen2.5-3B-Instruct"  # Llama, Phi, Gemma, SmolLM, …

# Voice — any text-to-speech model
tts_backend  = "huggingface"
tts_hf_repo  = "facebook/mms-tts-eng"      # SpeechT5, Bark, VITS, Parler, …
```

### Any Piper voice

Piper ships no voices. Name any of the ~180 in
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) and it
is fetched on first use into `~/.local/share/piper/voices`:

```toml
tts_backend = "piper"
tts_voice   = "en_GB-northern_english_male-medium"
# or "de_DE-thorsten-high", "fr_FR-siwis-medium", "ja_JA-hi_fi_captain-medium", …
# or the path of a .onnx you trained or downloaded yourself:
# tts_voice = "/home/me/voices/my-voice.onnx"

tts_download_voices = true   # false to stay offline and use only what is on disk
```

Names are `locale-speaker-quality`, and quality is one of `x_low`, `low`,
`medium`, `high` — bigger is better and slower. Try one without restarting
the orb:

```bash
nixorb tts "Testing this voice."
```

Gated repos need a token: set `hf_token`, or export `HF_TOKEN`.

### NVIDIA Nemotron 3.5 ASR (streaming)

[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a 0.6B FastConformer-RNNT covering 40 language-locales with punctuation and
capitalisation. It gets its own backend rather than going through the generic
pipeline because it is **cache-aware**: it reuses encoder state across strictly
non-overlapping chunks, so partial transcripts arrive *while you are still
talking* instead of after you stop.

```toml
asr_backend  = "nemotron"
asr_model    = "nvidia/nemotron-3.5-asr-streaming-0.6b"
asr_language = "auto"       # or a locale: "en-US", "de-DE", "ja-JP", …
asr_streaming = true        # emit partials as you speak

# Right-context in 80ms frames — the latency/accuracy dial.
asr_nemotron_lookahead = 3  # 0→80ms  3→320ms  6→560ms  13→1120ms
```

With `asr_language = "auto"` the model detects the spoken language and reports
it; NixOrb strips the `<en-US>` tag from the transcript and logs the detection.

Needs `transformers >= 5.13` — that is where `Nemotron3_5Asr` landed, and it
is the floor for the `[hf]` and `[nemotron]` extras. On an older install
NixOrb names the version it found and the command to fix it, then falls back
to offline transcription through the generic pipeline.

## Architecture Overview

![NixOrb Architecture Pipeline](docs/architecture_pipeline.png)

## Quick Start

### Install (Arch Linux)

```bash
git clone https://github.com/trail-b1az3r/nixorb.git
cd nixorb
chmod +x install.sh
./install.sh
```

The installer puts everything in a virtualenv at
`~/.local/share/nixorb/venv` and links the launcher into `~/.local/bin`.
Installing by hand elsewhere, do the same — see the note under
[Models](#models) for why a shared interpreter breaks this project:

```bash
python -m venv ~/.local/share/nixorb/venv
~/.local/share/nixorb/venv/bin/pip install torch torchaudio   # or the CUDA build
~/.local/share/nixorb/venv/bin/pip install -e .
ln -sf ~/.local/share/nixorb/venv/bin/nixorb ~/.local/bin/nixorb
```

Python 3.12 – 3.14 are supported.

### Start NixOrb

```bash
nixorb start          # Launch the floating orb
nixorb trigger        # Activate the running orb (bind to a KDE shortcut)
nixorb status         # Check system status
nixorb quit           # Shut the running orb down
nixorb ask "What is 2+2?"   # One-shot query
nixorb check          # Check dependencies
```

`nixorb trigger` talks to the running instance over a Unix socket in
`$XDG_RUNTIME_DIR`, so it works the same under Wayland, X11, or a bare TTY —
which is what makes it usable as a KDE global shortcut.

## Talking to it

It is built to be used by voice, not read.

- **Speaks while it thinks.** Each sentence is synthesised as the model
  produces it, so the reply starts about a second in rather than after the
  whole answer is written. (`tts_streaming`)
- **Talk over it.** Triggering while it is speaking cuts playback off and
  starts listening, instead of queueing behind the current answer.
  (`barge_in`)
- **Keep talking.** After answering it listens again for a few seconds, so
  "and what about tomorrow?" needs no second hotkey press.
  (`follow_up_seconds`, 0 to disable)
- **Answers made for ears.** The default prompt asks for one or two spoken
  sentences, answer first, no markdown — and shell commands are suppressed
  from the speech rather than read out character by character.
- **Live transcripts.** With Nemotron streaming on, partial text arrives
  while you are still speaking.

## Pipeline Flow

```
Trigger (Hotkey/WakeWord/Click)
  → Record Audio (sounddevice + VAD)
  → Transcribe (faster-whisper INT8, ~2.1GB VRAM)
  → Think (Ollama LLM stream, localhost:11434)
  → Speak (Piper TTS, offline)
  → Execute (<ACTION> sandboxed bash)
```

## VRAM Budget (GTX 1080 8 GB)

| Component | VRAM | Priority |
|-----------|------|----------|
| ASR (Whisper large-v3 INT8, or Nemotron 0.6B) | ~2.1 GB / ~1.3 GB | LOW — evicted first |
| LLM (Ollama, or an in-process HF model) | ~4.0 GB | HIGH |
| Piper TTS | ~0.1 GB | MEDIUM |
| System + KDE | ~0.5 GB | reserved |
| Safety buffer | 0.25 GB | reserved |

## Configuration

Edit `~/.config/nixorb/config.toml` or use the GUI (right-click orb → Settings):

```toml
hotkey = "Ctrl+Alt+Space"
orb_size = 88
llm_model = "llama3.2"
ollama_host = "http://localhost:11434"
tts_backend = "piper"

# Requires the optional extra: pip install 'nixorb[wakeword]'
wake_word_enabled = false

# Ask before running every command the model emits.
require_action_confirmation = true
```

See `config/default.toml` for every option with comments.

## Orb States

The orb has no caption — its colour is the state, and the tray tooltip spells
it out in words.

| State | Colour | Meaning |
|-------|--------|---------|
| Idle | Clay `#C96442` | Waiting for you |
| Listening | Sage `#5E9C7E` | Recording; the ring tracks your voice |
| Thinking | Amber `#D9A441` | Querying the model |
| Speaking | Lit clay `#E08A62` | Talking back |
| Error | Rust `#B3453E` | Something failed — check the log |

## Keyboard Shortcuts

| Action | Default | Notes |
|--------|---------|-------|
| Activate | `Ctrl+Alt+Space` | Global hotkey (via pynput/XWayland) |
| KDE Shortcut | Custom | System Settings → Shortcuts, running `nixorb trigger` |
| Orb double-click | Double-click | Direct activation |
| Orb right-click | Right-click | Menu: Activate / Settings / Quit |
| Opacity | Scroll wheel | While hovering over orb |
| Drag | Click+drag | Reposition the orb |

## Project Structure

```
nixorb/
├── core/           # Event bus, VRAM manager, IPC control socket
├── asr/            # base (mic + VAD) · faster-whisper · HF · Nemotron
├── llm/            # Ollama · Hugging Face · backend factory
├── tts/            # Piper · Hugging Face · streaming Speaker
├── ui/             # Qt6 orb, tray, settings, hotkey, confirm dialog
├── action/         # Confirmed command execution + clipboard
├── memory/         # ChromaDB vector memory
├── plugins/        # Plugin loader and tool dispatch
├── utils/          # Paths, logging, web search
├── vision/         # Screen capture
├── hf.py           # Shared Hugging Face device/dtype/auth plumbing
├── main.py         # Entry point and the conversation loop
├── cli.py          # CLI interface
└── settings.py     # Configuration
```

## Dependencies

- **System**: `python 3.12+`, `qt6-base`, `qt6-declarative`, `qt6-wayland`, `portaudio`, `wl-clipboard`, `grim`
- **LLM**: `ollama`, or `pip install 'nixorb[hf]'` for in-process models
- **AUR**: `piper-tts` (speech). Note its binary is `piper-tts` — Arch's
  `piper` package is the unrelated gaming-mouse tool. `espeak-ng` is the
  fallback if Piper is missing.
- **Python**: See `requirements.txt` / `pyproject.toml`

## Troubleshooting

NixOrb writes everything it does to `~/.local/share/nixorb/logs/nixorb.log`.
Start with `nixorb status` — it reports whether Ollama is actually reachable
and whether the configured model is installed.

| Symptom | Check |
|---|---|
| Orb never appears | `qt6-declarative` and `qt6-wayland` installed? The log prints any QML error. |
| `nixorb: command not found` | `~/.local/bin` on your `PATH` (the installer links it there). |
| Orb appears, nothing happens on activate | `nixorb status` — Ollama unreachable, or the model isn't pulled. |
| It speaks no audio | `yay -S piper-tts` (AUR), or install `espeak-ng`; the log says which is missing. |
| `nixorb trigger` says not running | Start the orb first; `nixorb status` shows the control socket. |
| Commands are always denied | Approve the confirmation dialog, or set `require_action_confirmation = false`. |
| A Hugging Face model won't load | `nixorb status` names the failure. Gated repo? Set `hf_token`. Custom architecture? `hf_trust_remote_code = true`. |
| Nemotron has no streaming | Needs `transformers >= 5.13`: `pip install 'nixorb[nemotron]'`. |
| pip prints `dependency conflicts` naming `transformers` | Another project in the same interpreter pins an older `transformers` than NixOrb's `>= 5.13`. They cannot share one; `nixorb check` names it. Put NixOrb in its own virtualenv. |
| It forgets everything between sessions | ChromaDB failed to load — `nixorb check`, then the log line starting `Memory:`. |
| `Turn failed: libcudart.so.12: cannot open shared object file` | torch and torchaudio came from different builds. `transformers>=5` imports torchaudio, so this kills model loading. Reinstall both from one index: `pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cpu` (or `…/whl/cu128`). `nixorb check` shows which build you have. |
| `Could not load this library: …/_torchaudio.so` | An earlier install left its extension behind and the loader finds two. `--force-reinstall` will not clear it: `pip uninstall -y torchaudio`, `rm -rf <site-packages>/torchaudio`, then reinstall. Or just leave it uninstalled — NixOrb never calls torchaudio and transformers skips it when absent. `nixorb check` lists the leftover files. |
| Speech recognition quietly switched to faster-whisper | The configured backend could not load; the log line starting `ASR:` says why. It keeps listening rather than failing every turn, and the decision is made once, not per turn. |
| `Failed to load model from file: …gguf` | The error now continues with what the file's own header says. "The file itself is fine … architecture 'x'" means the download is good and your llama.cpp is too old: `pip install -U --force-reinstall --no-cache-dir llama-cpp-python`. "Not a usable GGUF" means re-download. |
| `nemotron failed: expected string or bytes-like object, got 'list'` | Fixed in 2.0.11 — `generate()` returns a batched sequence and the decoder was handed the batch. Upgrade. |
| `Numba needs NumPy 2.4 or less` | numba (pulled in by the Nemotron stack) trails NumPy by a release. `pip install "numpy<2.5"`. NixOrb falls back to faster-whisper meanwhile. |
| Piper only ever uses one voice | Fixed in 2.0.13 — before that it searched disk only, so any voice the installer had not fetched fell back to espeak. Set `tts_voice` to any name from `rhasspy/piper-voices`. |
| It speaks with espeak although `tts_backend = "piper"` | The log line starting `TTS:` says why: no `piper-tts` binary, or the voice could not be fetched. A `tts_voice` written as prose (the stock default, meant for the HF backend) now falls back to `en_US-lessac-medium` rather than to espeak. |
| Custom wake word ignored | `wake_word_model` must be an openwakeword `.onnx`/`.tflite` path, or one of its bundled names. A Hugging Face repo of a transformers audio classifier is a different kind of model and openwakeword cannot load it — train one with openwakeword's tools instead. |
| Out of VRAM with an HF LLM | `llm_hf_load_in_4bit = true` (`pip install 'nixorb[quant]'`), or `hf_device = "cpu"`. |
| Actions do nothing when run as root | NixOrb disables command execution as root — run it as your normal user. |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT © NixOrb Contributors
