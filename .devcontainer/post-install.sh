#!/bin/bash

set -e

# Configure git user for reproducible commit attribution
git config --global user.name "OrNot"
git config --global user.email "bin.li@kickerlab.com"

echo "� TrackLab post-install setup..."

# Activate virtual environment
source .venv/bin/activate

# Set pip index to Chinese mirror for faster downloads
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Increase UV HTTP timeout to handle slow downloads
export UV_HTTP_TIMEOUT=1200

# Install PyTorch with CUDA support (matching current working environment)
echo "📦 Installing PyTorch with CUDA..."
uv pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 --force-reinstall

# Downgrade NumPy to fix compatibility issues with imgaug and other libraries
echo "📦 Downgrading NumPy to version < 2.0 for compatibility..."
uv pip install "numpy<2.0" --force-reinstall --no-cache-dir

# Install compatible versions to prevent conflicts during uv sync
echo "📦 Installing compatible dependency versions..."
uv pip install monai==1.0.1 transformers==4.22.2 scikit-learn --force-reinstall --no-cache-dir

# Install build dependencies for MMCV compilation
echo "🔧 Installing build dependencies for MMCV..."
apt-get update && apt-get install -y build-essential python3-dev

# Run uv sync to install most dependencies (excluding mmcv to avoid CPU-only version)
echo "📦 Installing main project dependencies via uv sync..."
uv sync --no-install-package mmcv || {
    echo "  ⚠️  uv sync failed, trying fallback..."
    uv pip install -e . || echo "  ⚠️  Failed to install main dependencies"
}

# Install MMCV with CUDA support from OpenMMLab (CRITICAL for _ext module)
echo "📦 Installing MMCV 2.0.1 with CUDA 11.7 support..."
pip uninstall mmcv mmcv-full -y 2>/dev/null || true
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --no-cache-dir --force-reinstall

# Verify MMCV _ext module works
echo "🔍 Verifying MMCV _ext module..."
python -c 'import mmcv._ext; print("✅ MMCV _ext imported successfully")' || {
    echo "❌ MMCV _ext import failed - container may need CUDA runtime"
    exit 1
}

# Install development dependencies
echo "📦 Installing development dependencies..."
uv sync --group dev || {
    echo "  ⚠️  uv dev sync failed, trying uv pip install..."
    uv pip install sphinx sphinx_rtd_theme myst-parser || echo "  ⚠️  Some dev dependencies may have failed to install"
}

# Install prtreid (only if not already installed)
echo "🔧 Installing prtreid..."
if python -c "import prtreid" 2>/dev/null; then
    echo "  ✅ prtreid already installed, skipping"
else
    echo "  📦 Installing prtreid and its dependencies..."
    uv pip install git+https://github.com/VlSomers/prtreid.git || {
        echo "  ⚠️  Standard installation failed, trying with --no-deps..."
        uv pip install git+https://github.com/VlSomers/prtreid.git --no-deps || echo "  ⚠️  Failed to install prtreid, you may need to install it manually"
    }
fi

# Install bpbreid (only if not already installed)
echo "🔧 Installing bpbreid..."
if python -c "import torchreid" 2>/dev/null; then
    echo "  ✅ bpbreid already installed, skipping"
else
    echo "  📦 Installing tensorboard (required by bpbreid)..."
    uv pip install tensorboard
    echo "  📦 Installing bpbreid (with --no-deps to avoid tb-nightly issue)..."
    uv pip install git+https://github.com/VlSomers/bpbreid.git --no-deps || echo "  ⚠️  Failed to install bpbreid, you may need to install it manually"
fi

# Verify key dependencies are installed and working
echo "🔍 Verifying key dependencies..."
python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"
python -c "import monai; print('✅ monai:', monai.__version__)" 2>/dev/null || echo "❌ monai not found"
python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers not found"
python -c "import mmcv; print('✅ mmcv:', mmcv.__version__)" 2>/dev/null || echo "❌ mmcv not found"
python -c "import mmocr; print('✅ mmocr:', mmocr.__version__)" 2>/dev/null || echo "❌ mmocr not found"

echo "Testing TrackLab imports..."
python -c "from tracklab.pipeline.jersey.mmocr_api import MMOCR; from tracklab.pipeline.reid.prtreid_api import PRTReId; print('✅ All TrackLab imports successful!')" 2>/dev/null || {
    echo "❌ TrackLab imports failed!"
    exit 1
}

# Final verification - test TrackLab command
echo "🔍 Testing TrackLab command..."
tracklab --help > /dev/null 2>&1 && echo "✅ TrackLab command works!" || {
    echo "❌ TrackLab command failed!"
    exit 1
}

echo "🎉 Devcontainer setup complete and verified!"
