#!/bin/bash#!/bin/bash

set -eset -e



echo "🔧 TrackLab post-install setup..."echo "# Install compatible versions to prevent conflicts during uv sync

echo "📦 Installing compatible dependency versions..."

# Activate virtual environmentuv pip install monai==1.0.1 transformers==4.22.2 --force-reinstall --no-cache-dir

source .venv/bin/activate

# Run uv sync to install most dependencies (excluding mmcv to avoid CPU-only version)

# Set pip index to Chinese mirror for faster downloadsecho "📦 Installing main project dependencies via uv sync..."

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simpleuv sync --no-install-package mmcv || {

export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn    echo "  ⚠️  uv sync failed, trying fallback..."

    uv pip install -e . || echo "  ⚠️  Failed to install main dependencies"

# Increase UV HTTP timeout to handle slow downloads}

export UV_HTTP_TIMEOUT=600

# Install MMCV with CUDA support from OpenMMLab (CRITICAL for _ext module)

# Install PyTorch with CUDA support (matching current working environment)echo "📦 Installing MMCV 2.0.1 with CUDA 11.7 support..."

echo "📦 Installing PyTorch with CUDA..."pip uninstall mmcv mmcv-full -y 2>/dev/null || true

uv pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 --force-reinstallpip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --no-cache-dir --force-reinstall



# Downgrade NumPy to fix compatibility issues with imgaug and other libraries# Verify MMCV _ext module works

echo "📦 Downgrading NumPy to version < 2.0 for compatibility..."echo "🔍 Verifying MMCV _ext module..."

uv pip install "numpy<2.0" --force-reinstall --no-cache-dirpython -c 'import mmcv._ext; print("✅ MMCV _ext imported successfully")' || {

    echo "❌ MMCV _ext import failed - container may need CUDA runtime"

# Install compatible versions to prevent conflicts during uv sync    exit 1

echo "📦 Installing compatible dependency versions..."}ost-install setup..."

uv pip install monai==1.0.1 transformers==4.22.2 --force-reinstall --no-cache-dir

# Activate virtual environment

# Install build dependencies for MMCV compilationsource .venv/bin/activate

echo "🔧 Installing build dependencies for MMCV..."

apt-get update && apt-get install -y build-essential python3-dev# Set pip index to Chinese mirror for faster downloads

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Run uv sync to install most dependencies (excluding mmcv to avoid CPU-only version)export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "📦 Installing main project dependencies via uv sync..."

uv sync --no-install-package mmcv || {# Increase UV HTTP timeout to handle slow downloads

    echo "  ⚠️  uv sync failed, trying fallback..."export UV_HTTP_TIMEOUT=600

    uv pip install -e . || echo "  ⚠️  Failed to install main dependencies"

}# Install PyTorch with CUDA support (matching current working environment)

echo "📦 Installing PyTorch with CUDA..."

# Install MMCV with CUDA support from OpenMMLab (CRITICAL for _ext module)uv pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 --force-reinstall

echo "📦 Installing MMCV 2.0.1 with CUDA 11.7 support..."

pip uninstall mmcv mmcv-full -y 2>/dev/null || true# Downgrade NumPy to fix compatibility issues with imgaug and other libraries

pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --no-cache-dir --force-reinstallecho "📦 Downgrading NumPy to version < 2.0 for compatibility..."

uv pip install "numpy<2.0" --force-reinstall --no-cache-dir

# Verify MMCV _ext module works

echo "🔍 Verifying MMCV _ext module..."# Install compatible versions BEFORE running uv sync to prevent conflicts

python -c 'import mmcv._ext; print("✅ MMCV _ext imported successfully")' || {echo "📦 Installing compatible dependency versions..."

    echo "❌ MMCV _ext import failed - container may need CUDA runtime"uv pip install monai==1.0.1 transformers==4.22.2 --force-reinstall --no-cache-dir

    exit 1

}# Install compatible versions to prevent conflicts during uv sync

echo "📦 Installing compatible dependency versions..."

# Install development dependenciesuv pip install monai==1.0.1 transformers==4.22.2 --force-reinstall --no-cache-dir

echo "📦 Installing development dependencies..."

uv sync --group dev || {# Run uv sync exactly as it was when requirements.txt was created

    echo "  ⚠️  uv dev sync failed, trying uv pip install..."echo "📦 Installing all dependencies via uv sync (recreating working state)..."

    uv pip install sphinx sphinx_rtd_theme myst-parser || echo "  ⚠️  Some dev dependencies may have failed to install"uv sync

}

# Verify MMCV installation

# Install prtreid (only if not already installed)echo "� Verifying MMCV installation..."

echo "🔧 Installing prtreid..."python -c "import mmcv; print('✅ MMCV version:', mmcv.__version__)"

if python -c "import prtreid" 2>/dev/null; thenpython -c "import mmcv._ext; print('✅ MMCV _ext imported successfully')" 2>/dev/null || echo "❌ MMCV _ext import failed"

    echo "  ✅ prtreid already installed, skipping"

else# Install main project dependencies from pyproject.toml (but don't let it override our working versions)

    echo "  📦 Installing prtreid and its dependencies..."echo "📦 Installing main project dependencies..."

    uv pip install git+https://github.com/VlSomers/prtreid.git || {uv sync --no-install-package torch --no-install-package torchvision --no-install-package monai --no-install-package transformers --no-install-package mmocr --no-install-package mmdet --no-install-package mmcv || {

        echo "  ⚠️  Standard installation failed, trying with --no-deps..."    echo "  ⚠️  uv sync failed, trying uv pip install..."

        uv pip install git+https://github.com/VlSomers/prtreid.git --no-deps || echo "  ⚠️  Failed to install prtreid, you may need to install it manually"    uv pip install -e . || echo "  ⚠️  Failed to install main dependencies"

    }}

fi

# Install development dependencies

# Install bpbreid (only if not already installed)echo "📦 Installing development dependencies..."

# Note: bpbreid requires tb-nightly which is not available, so we install tensorboard firstuv sync --group dev || {

echo "🔧 Installing bpbreid..."    echo "  ⚠️  uv dev sync failed, trying uv pip install..."

if python -c "import torchreid" 2>/dev/null; then    uv pip install sphinx sphinx_rtd_theme myst-parser || echo "  ⚠️  Some dev dependencies may have failed to install"

    echo "  ✅ bpbreid already installed, skipping"}

else

    echo "  📦 Installing tensorboard (required by bpbreid)..."# Install prtreid (only if not already installed)

    uv pip install tensorboardecho "🔧 Installing prtreid..."

    echo "  📦 Installing bpbreid (with --no-deps to avoid tb-nightly issue)..."if python -c "import prtreid" 2>/dev/null; then

    uv pip install git+https://github.com/VlSomers/bpbreid.git --no-deps || echo "  ⚠️  Failed to install bpbreid, you may need to install it manually"    echo "  ✅ prtreid already installed, skipping"

fielse

    echo "  📦 Installing prtreid and its dependencies..."

# Verify key dependencies are installed and working    uv pip install git+https://github.com/VlSomers/prtreid.git || {

echo "🔍 Verifying key dependencies..."        echo "  ⚠️  Standard installation failed, trying with --no-deps..."

echo "Checking PyTorch..."        uv pip install git+https://github.com/VlSomers/prtreid.git --no-deps || echo "  ⚠️  Failed to install prtreid, you may need to install it manually"

python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"    }

fi

echo "Checking MONAI..."

python -c "import monai; print('✅ monai:', monai.__version__)" 2>/dev/null || echo "❌ monai not found"# Install bpbreid (only if not already installed)

# Note: bpbreid requires tb-nightly which is not available, so we install tensorboard first

echo "Checking Transformers..."echo "🔧 Installing bpbreid..."

python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers not found"if python -c "import torchreid" 2>/dev/null; then

    echo "  ✅ bpbreid already installed, skipping"

echo "Checking MMCV..."else

python -c "import mmcv; print('✅ mmcv:', mmcv.__version__)" 2>/dev/null || echo "❌ mmcv not found"    echo "  📦 Installing tensorboard (required by bpbreid)..."

    uv pip install tensorboard

echo "Checking MMOCR..."    echo "  📦 Installing bpbreid (with --no-deps to avoid tb-nightly issue)..."

python -c "import mmocr; print('✅ mmocr:', mmocv.__version__)" 2>/dev/null || echo "❌ mmocr not found"    uv pip install git+https://github.com/VlSomers/bpbreid.git --no-deps || echo "  ⚠️  Failed to install bpbreid, you may need to install it manually"

fi

echo "Testing TrackLab imports..."

python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; from tracklab.wrappers.reid.prtreid_api import PRTReId; print('✅ All TrackLab imports successful!')" 2>/dev/null || {# Verify key dependencies are installed and working

    echo "❌ TrackLab imports failed!"echo "🔍 Verifying key dependencies..."

    exit 1echo "Checking PyTorch..."

}python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"



# Final verification - test TrackLab commandecho "Checking MONAI..."

echo "🔍 Testing TrackLab command..."python -c "import monai; print('✅ monai:', monai.__version__)" 2>/dev/null || echo "❌ monai not found"

tracklab --help > /dev/null 2>&1 && echo "✅ TrackLab command works!" || {

    echo "❌ TrackLab command failed!"echo "Checking Transformers..."

    exit 1python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers not found"

}

echo "Checking MMCV..."

echo "🎉 Devcontainer setup complete and verified!"python -c "import mmcv; print('✅ mmcv:', mmcv.__version__)" 2>/dev/null || echo "❌ mmcv not found"

python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; print('✅ MMOCR API imported successfully')" 2>/dev/null || echo "❌ MMOCR API failed"

python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"echo "Checking MMOCR..."

python -c "import torchvision; print('✅ torchvision:', torchvision.__version__)" 2>/dev/null || echo "❌ torchvision not found"python -c "import mmocr; print('✅ mmocr:', mmocr.__version__)" 2>/dev/null || echo "❌ mmocr not found"

python -c "import prtreid; print('✅ prtreid imported successfully')" 2>/dev/null || echo "❌ prtreid not found"

python -c "import torchreid; print('✅ torchreid imported successfully')" 2>/dev/null || echo "❌ torchreid not found"echo "Testing TrackLab imports..."

python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; from tracklab.wrappers.reid.prtreid_api import PRTReId; print('✅ All TrackLab imports successful!')" 2>/dev/null || {

echo "🎉 Devcontainer setup complete!"    echo "❌ TrackLab imports failed!"
    exit 1
}

echo "🎉 Devcontainer setup complete and verified!"
python -c "from tracklab.wrappers.jersey.mmocr_api import MMOCR; print('✅ MMOCR API imported successfully')" 2>/dev/null || echo "❌ MMOCR API failed"
python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch not found"
python -c "import torchvision; print('✅ torchvision:', torchvision.__version__)" 2>/dev/null || echo "❌ torchvision not found"
python -c "import prtreid; print('✅ prtreid imported successfully')" 2>/dev/null || echo "❌ prtreid not found"
python -c "import torchreid; print('✅ torchreid imported successfully')" 2>/dev/null || echo "❌ torchreid not found"

echo "🎉 Devcontainer setup complete!"
