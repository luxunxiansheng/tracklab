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
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 --force-reinstall

# Downgrade NumPy to fix compatibility issues with imgaug and other libraries
echo "📦 Downgrading NumPy to version < 2.0 for compatibility..."
pip install "numpy<2.0" --force-reinstall --no-cache-dir

# Install compatible versions BEFORE running uv sync to prevent conflicts
echo "📦 Installing compatible dependency versions..."
pip install monai==1.0.1 transformers==4.22.2 --force-reinstall --no-cache-dir

# Install MMCV ecosystem in correct order
echo "📦 Installing MMCV 2.0.1 with CUDA 11.7 support (includes _ext module)..."
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --force-reinstall --no-cache-dir

echo "📦 Installing MMDetection 3.0.0..."
pip install "mmdet>=3.0.0rc0,<3.1.0" --force-reinstall --no-cache-dir

echo "📦 Installing MMOCR 1.0.1..."
pip install mmocr==1.0.1 --force-reinstall --no-cache-dir

# Install main project dependencies from pyproject.toml (but don't let it override our working versions)
echo "📦 Installing main project dependencies..."
uv sync --no-install-package torch --no-install-package torchvision --no-install-package monai --no-install-package transformers --no-install-package mmocr --no-install-package mmdet --no-install-package mmcv || {
    echo "  ⚠️  uv sync failed, trying pip install..."
    pip install -e . || echo "  ⚠️  Failed to install main dependencies"
}

# Install development dependencies
echo "📦 Installing development dependencies..."
uv sync --group dev || {
    echo "  ⚠️  uv dev sync failed, trying pip install..."
    pip install sphinx sphinx_rtd_theme myst-parser || echo "  ⚠️  Some dev dependencies may have failed to install"
}

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

# Verify key dependencies are installed and working
echo "🔍 Verifying key dependencies..."
echo "Checking PyTorch..."
python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"

echo "Checking MONAI..."
python -c "import monai; print('✅ monai:', monai.__version__)" 2>/dev/null || echo "❌ monai not found"

echo "Checking Transformers..."
python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers not found"

echo "Checking MMCV..."
python -c "import mmcv; print('✅ mmcv:', mmcv.__version__)" 2>/dev/null || echo "❌ mmcv not found"

echo "Checking MMOCR..."
python -c "import mmocr; print('✅ mmocr:', mmocr.__version__)" 2>/dev/null || echo "❌ mmocr not found"

echo "Testing TrackLab imports..."
python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; from tracklab.wrappers.reid.prtreid_api import PRTReId; print('✅ All TrackLab imports successful!')" 2>/dev/null || {
    echo "❌ TrackLab imports failed!"
    exit 1
}

echo "🎉 Devcontainer setup complete and verified!"
python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; print('✅ MMOCR API imported successfully')" 2>/dev/null || echo "❌ MMOCR API failed"
python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"
python -c "import torchvision; print('✅ torchvision:', torchvision.__version__)" 2>/dev/null || echo "❌ torchvision not found"
python -c "import prtreid; print('✅ prtreid imported successfully')" 2>/dev/null || echo "❌ prtreid not found"
python -c "import torchreid; print('✅ torchreid imported successfully')" 2>/dev/null || echo "❌ torchreid not found"

echo "🎉 Devcontainer setup complete!"
