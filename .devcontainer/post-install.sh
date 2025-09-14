#!/bin/bash
set -e

echo "🔧 TrackLab post-install setup..."

# Activate virtual environment
source .venv/bin/activate

# Set pip index to Chinese mirror for faster downloads
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Install PyTorch with CUDA support (matching current working environment)
echo "📦 Installing PyTorch with CUDA..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install main project dependencies from pyproject.toml
echo "📦 Installing main project dependencies..."
uv sync || {
    echo "  ⚠️  uv sync failed, trying pip install..."
    pip install -e . || echo "  ⚠️  Failed to install main dependencies"
}

# Install development dependencies
echo "📦 Installing development dependencies..."
uv sync --group dev || {
    echo "  ⚠️  uv dev sync failed, trying pip install..."
    pip install sphinx sphinx_rtd_theme myst-parser || echo "  ⚠️  Some dev dependencies may have failed to install"
}

# Install additional dependencies that may be missing
echo "📦 Installing additional dependencies..."
pip install monai torchmetrics || echo "  ⚠️  Some additional dependencies may have failed to install"

# Install prtreid (only if not already installed)
echo "🔧 Installing prtreid..."
if python -c "import prtreid" 2>/dev/null; then
    echo "  ✅ prtreid already installed, skipping"
else
    echo "  📦 Installing prtreid and its dependencies..."
    pip install git+https://github.com/VlSomers/prtreid.git || {
        echo "  ⚠️  Standard installation failed, trying with --no-deps..."
        pip install git+https://github.com/VlSomers/prtreid.git --no-deps || echo "  ⚠️  Failed to install prtreid, you may need to install it manually"
    }
fi

# Install bpbreid (only if not already installed)
# Note: bpbreid requires tb-nightly which is not available, so we install tensorboard first
echo "🔧 Installing bpbreid..."
if python -c "import torchreid" 2>/dev/null; then
    echo "  ✅ bpbreid already installed, skipping"
else
    echo "  📦 Installing tensorboard (required by bpbreid)..."
    pip install tensorboard
    echo "  📦 Installing bpbreid (with --no-deps to avoid tb-nightly issue)..."
    pip install git+https://github.com/VlSomers/bpbreid.git --no-deps || echo "  ⚠️  Failed to install bpbreid, you may need to install it manually"
fi



# Fix to
