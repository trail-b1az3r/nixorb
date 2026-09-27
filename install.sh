#!/bin/bash
# NixOrb installer for Arch Linux + KDE Plasma 6
# Usage: ./install.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      NixOrb Installer for Arch Linux     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# Check if running on Arch
if ! command -v pacman &> /dev/null; then
    echo -e "${RED}Error: This installer is for Arch Linux only.${NC}"
    echo "For other distributions, install dependencies manually."
    exit 1
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✓ NVIDIA GPU detected: $GPU_INFO${NC}"
    HAS_NVIDIA=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected — CPU inference will be slower${NC}"
    HAS_NVIDIA=false
fi

# ── System Dependencies ────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing system dependencies…${NC}"

sudo pacman -S --needed --noconfirm \
    python python-pip python-virtualenv \
    qt6-base qt6-declarative qt6-wayland qt6-multimedia \
    portaudio wl-clipboard grim slurp \
    espeak-ng bubblewrap \
    git base-devel

echo -e "${GREEN}✓ System dependencies installed${NC}"

# ── Piper TTS (AUR) ────────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing Piper TTS…${NC}"

# The AUR piper-tts package installs its binary as `piper-tts`. Arch's own
# `piper` package is the gaming-mouse tool, so never test for plain `piper`
# first — you would "find" completely unrelated software.
if command -v piper-tts &> /dev/null; then
    echo -e "${GREEN}✓ piper-tts already installed${NC}"
else
    echo "piper-tts not found — installing from the AUR"
    AUR_HELPER=""
    command -v yay  &> /dev/null && AUR_HELPER="yay"
    [ -z "$AUR_HELPER" ] && command -v paru &> /dev/null && AUR_HELPER="paru"

    if [ -n "$AUR_HELPER" ]; then
        if "$AUR_HELPER" -S --needed --noconfirm piper-tts; then
            echo -e "${GREEN}✓ piper-tts installed${NC}"
        else
            echo -e "${YELLOW}⚠ piper-tts failed to build — falling back to espeak-ng.${NC}"
            echo "   Retry later with: $AUR_HELPER -S piper-tts"
        fi
    else
        echo -e "${YELLOW}⚠ No AUR helper found (install yay or paru).${NC}"
        echo "   Then run:  yay -S piper-tts"
        echo "   Until then NixOrb speaks through espeak-ng."
    fi
fi

# ── Ollama ─────────────────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing Ollama…${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
else
    echo -e "${GREEN}✓ Ollama already installed${NC}"
fi

# Start the Ollama service. The official installer creates a *system* unit,
# so `systemctl --user` fails on most setups; try the system unit first and
# fall back to a background `ollama serve`.
if ! curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama…"
    if systemctl list-unit-files ollama.service >/dev/null 2>&1; then
        sudo systemctl enable --now ollama || true
    elif systemctl --user list-unit-files ollama.service >/dev/null 2>&1; then
        systemctl --user enable --now ollama || true
    else
        echo -e "${YELLOW}⚠ No ollama service unit found.${NC}"
        echo "   Start it manually in another terminal:  ollama serve"
    fi

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

if curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is responding on localhost:11434${NC}"
else
    echo -e "${YELLOW}⚠ Ollama is not responding — NixOrb will start but cannot think.${NC}"
    echo "   Run 'ollama serve' and then 'nixorb status' to re-check."
fi

# ── Python Environment ─────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Setting up Python environment…${NC}"

VENV_DIR="${HOME}/.local/share/nixorb/venv"
mkdir -p "$VENV_DIR"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# torch and torchaudio ALWAYS come from one index. Mixing them — a CUDA
# torchaudio beside a CPU torch, say — installs cleanly and then fails at
# the dynamic linker the first time a model loads, with a message naming
# only a missing .so ("libcudart.so.12: cannot open shared object file")
# and nothing about torch. Note the CPU branch pins the CPU index too:
# a bare `pip install torch` on Linux pulls the CUDA build.
#
# The cu118 pin is this project's GTX 1080 target, but those wheels stop at
# cp313 — torch 2.7.1 has no 3.14 build — so newer interpreters take the
# current CUDA 12.8 release instead of failing here and leaving the user to
# install torch by hand, which is how the halves come to disagree.
PY_MINOR=$(python -c 'import sys; print(sys.version_info[1])')

if [ "$HAS_NVIDIA" = true ] && [ "$PY_MINOR" -le 13 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_SPEC="torch==2.7.1+cu118 torchaudio==2.7.1+cu118"
elif [ "$HAS_NVIDIA" = true ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_SPEC="torch torchaudio"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_SPEC="torch torchaudio"
fi

echo -e "${BLUE}→ Installing PyTorch from ${TORCH_INDEX}…${NC}"
pip install $TORCH_SPEC --index-url "$TORCH_INDEX"

# llama-cpp-python compiles from source by default (several minutes). The
# model this project ships with is tiny (~1.3 GB Q4_K_M GGUF), so CPU
# inference is plenty fast — grab the prebuilt CPU wheel instead of building.
echo -e "${BLUE}→ Installing llama-cpp-python (prebuilt CPU wheel)…${NC}"
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install NixOrb
echo -e "${BLUE}→ Installing NixOrb…${NC}"
pip install -e "."

# ── Put `nixorb` on PATH ───────────────────────────────────────── #
# Everything above installs into a venv that is not on PATH, so without
# this `nixorb start` is "command not found" the moment the shell that ran
# this script exits.
echo -e "${BLUE}→ Linking nixorb into ~/.local/bin…${NC}"
mkdir -p "${HOME}/.local/bin"
ln -sf "$VENV_DIR/bin/nixorb" "${HOME}/.local/bin/nixorb"

case ":$PATH:" in
    *":${HOME}/.local/bin:"*) ;;
    *)
        echo -e "${YELLOW}⚠ ~/.local/bin is not on your PATH.${NC}"
        echo "   Add this to ~/.bashrc or ~/.config/fish/config.fish:"
        echo "     export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
esac

# ── Pull default model ─────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Pulling default LLM model (llama3.2)…${NC}"
if ! ollama pull llama3.2; then
    echo -e "${YELLOW}⚠ Could not pull llama3.2 — run 'ollama pull llama3.2' later.${NC}"
fi

# ── Piper voice model ──────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Setting up Piper voice model…${NC}"

VOICE="en_US-lessac-medium"
VOICE_DIR="${HOME}/.local/share/piper/voices"

# NixOrb searches the AUR piper-voices layout too, so if the package is
# already installed there is nothing to download.
if find /usr/share/piper-voices -name "${VOICE}.onnx" 2>/dev/null | grep -q . ; then
    echo -e "${GREEN}✓ ${VOICE} provided by the piper-voices package${NC}"
elif [ -f "$VOICE_DIR/${VOICE}.onnx" ]; then
    echo -e "${GREEN}✓ Voice model already downloaded${NC}"
else
    mkdir -p "$VOICE_DIR"
    BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
    if curl -fL -o "$VOICE_DIR/${VOICE}.onnx"      "${BASE}/${VOICE}.onnx" \
    && curl -fL -o "$VOICE_DIR/${VOICE}.onnx.json" "${BASE}/${VOICE}.onnx.json"; then
        echo -e "${GREEN}✓ Voice model downloaded${NC}"
    else
        rm -f "$VOICE_DIR/${VOICE}.onnx" "$VOICE_DIR/${VOICE}.onnx.json"
        echo -e "${YELLOW}⚠ Voice download failed — NixOrb will use espeak-ng.${NC}"
        echo "   Or install the AUR package:  yay -S piper-voices-en-us"
    fi
fi

# ── Desktop Entry ──────────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Creating desktop entry…${NC}"

mkdir -p "${HOME}/.local/share/applications"
cat > "${HOME}/.local/share/applications/nixorb.desktop" << 'EOF'
[Desktop Entry]
Name=NixOrb
Comment=Floating AI Assistant
Exec=nixorb start
Icon=audio-input-microphone
Type=Application
Categories=Utility;Audio;
StartupNotify=false
EOF

echo -e "${GREEN}✓ Desktop entry created${NC}"

# ── Configure for this machine ─────────────────────────────────── #
# The shipped defaults aim high. This looks at what is actually
# installed and writes settings that will run here, rather than leaving
# the first launch to fail on a model this machine cannot load.
echo ""
echo -e "${BLUE}→ Configuring NixOrb for this machine…${NC}"
"$VENV_DIR/bin/nixorb" setup --yes || {
    echo -e "${YELLOW}⚠ Automatic setup failed — run 'nixorb setup' yourself.${NC}"
}

# ── KDE Shortcut ───────────────────────────────────────────────── #
echo ""
echo -e "${YELLOW}⚠ Important: Set up KDE shortcut:${NC}"
echo "   1. Open System Settings → Shortcuts → Custom Shortcuts"
echo "   2. Add a new global shortcut → Command/URL"
echo "   3. Set trigger: Meta+Space (or your preference)"
echo "   4. Set command: nixorb trigger"
echo "   5. Apply and save"
echo ""

# ── Done ───────────────────────────────────────────────────────── #
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     NixOrb installation complete!        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Start NixOrb:     nixorb start"
echo "Check status:     nixorb status"
echo "Edit config:      nixorb config"
echo "Check deps:       nixorb check"
echo ""
echo -e "${BLUE}Logs: ~/.local/share/nixorb/logs/nixorb.log${NC}"
echo ""
