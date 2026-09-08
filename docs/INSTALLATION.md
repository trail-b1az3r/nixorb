# NixOrb Installation Guide

## Prerequisites

- Arch Linux (rolling release)
- KDE Plasma 6 + Wayland
- NVIDIA GPU (tested: GTX 1080 8 GB)
- Python 3.12+

## Quick Install (Recommended)

```bash
git clone https://github.com/trail-b1az3r/nixorb.git
cd nixorb
bash scripts/setup_dev.sh
```

## Manual Install

### 1 — System packages

```bash
sudo pacman -S --needed \
    python qt6-base qt6-declarative qt6-wayland \
    qt6-multimedia qt6-tools \
    cuda cudnn portaudio wl-clipboard grim ffmpeg base-devel
```

### 2 — Python (fish shell)

```fish
python -m venv .venv --system-site-packages
source .venv/bin/activate.fish
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install piper-tts openwakeword
```

### 3 — Compile shaders

```bash
bash scripts/compile_shaders.sh
# Or manually: /usr/lib/qt6/bin/qsb --glsl "100es,120,150" --hlsl 50 --msl 12 \
#   assets/shaders/orb_glow.vert -o assets/shaders/orb_glow.vert.qsb
```

### 4 — First run

```bash
nixorb config hf_token hf_xxxxx   # for gated HF models
nixorb start
```

## Wayland + Hotkey Notes

Global hotkeys use **pynput** which requires XWayland.
NixOrb auto-detects `DISPLAY` from `/tmp/.X11-unix/`.

If hotkeys don't work:
```bash
echo $DISPLAY      # should be :0 or :1
export DISPLAY=:0  # if unset
```

Or double-click the orb to trigger, or use the tray icon.

## GPU Setup

```bash
sudo pacman -S cuda cudnn
nvidia-smi   # verify GPU visible
nixorb version   # shows VRAM info
```
