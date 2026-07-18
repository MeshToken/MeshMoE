#!/usr/bin/env bash
# MeshMoE Node — one-line installer for Linux/macOS
# Usage: curl -fsSL https://meshmoe.com/install.sh | bash
# Or:   wget -qO- https://meshmoe.com/install.sh | bash
set -e

# Colors
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
echo -e "${G}╔══════════════════════════════════════════╗${N}"
echo -e "${G}║   MeshMoE Node Installer                  ║${N}"
echo -e "${G}║   Your computer. The world's AI.          ║${N}"
echo -e "${G}╚══════════════════════════════════════════╝${N}"
echo

INSTALL_DIR="${MESHMOE_HOME:-$HOME/.meshmoe}"

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${R}✗ Python 3 not found.${N} Install Python 3.10+ first:"
    echo "  macOS:  brew install python"
    echo "  ubuntu: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
PYVER=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo -e "${G}✓ Python $PYVER${N}"

# 2. Check GPU (optional but recommended)
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${G}✓ GPU detected: $GPU${N}"
else
    echo -e "${Y}⚠ No NVIDIA GPU detected. You can still install but inference will be CPU-only (slow).${N}"
fi

# 3. Clone repo
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${Y}→ Existing install at $INSTALL_DIR, updating...${N}"
    cd "$INSTALL_DIR" && git pull --rebase || true
else
    echo "→ Cloning to $INSTALL_DIR"
    git clone https://github.com/OpenMeshMoE/MeshMoE.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 4. Create venv
echo "→ Creating Python virtual environment..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

# 5. Install deps
echo "→ Installing dependencies (llama-cpp-python builds, may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt

# 6. Done
echo
echo -e "${G}╔══════════════════════════════════════════╗${N}"
echo -e "${G}║   ✓ Install complete!                     ║${N}"
echo -e "${G}╚══════════════════════════════════════════╝${N}"
echo
echo "Next steps:"
echo
echo "  1. Sign in at https://meshmoe.com/app/ and create an API key"
echo
echo "  2. cd $INSTALL_DIR"
echo "     source venv/bin/activate"
echo
echo "  3. Pick your expert model and run:"
echo "       python edge_node.py --model glm-4-9b              # 12GB VRAM"
echo "       python edge_node.py --model deepseek-r1-distill-qwen-14b   # 16GB+ (recommended)"
echo
echo "  4. Watch your earnings at https://meshmoe.com/app/earnings"
echo
echo "Audit the client before long-term use: https://github.com/OpenMeshMoE/MeshMoE"
echo
