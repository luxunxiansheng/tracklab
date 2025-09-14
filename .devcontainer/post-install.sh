#!/bin/bash
set -e

echo "🔧 TrackLab post-install setup..."

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA support (matching current working environment)
echo "📦 Installing PyTorch with CUDA..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install prtreid (only if not already installed)
echo "🔧 Installing prtreid..."
if python -c "import prtreid" 2>/dev/null; then
    echo "  ✅ prtreid already installed, skipping"
else
    pip install git+https://github.com/VlSomers/prtreid.git --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install git+https://github.com/VlSomers/prtreid.git --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# Install bpbreid (only if not already installed)
echo "🔧 Installing bpbreid..."
if python -c "import torchreid" 2>/dev/null; then
    echo "  ✅ bpbreid already installed, skipping"
else
    pip install git+https://github.com/VlSomers/bpbreid.git --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install git+https://github.com/VlSomers/bpbreid.git --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# Install optional dependencies to avoid warnings
echo "📦 Installing optional dependencies..."

# Core scientific packages
if python -c "import scipy" 2>/dev/null; then
    echo "  ✅ scipy already installed, skipping"
else
    pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ scipy install skipped"
fi

if python -c "import sklearn" 2>/dev/null; then
    echo "  ✅ scikit-learn already installed, skipping"
else
    pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ scikit-learn install skipped"
fi

if python -c "import PIL" 2>/dev/null; then
    echo "  ✅ pillow already installed, skipping"
else
    pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ pillow install skipped"
fi

# Computer vision packages
if python -c "import cv2" 2>/dev/null; then
    echo "  ✅ opencv-python already installed, skipping"
else
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ opencv-python install skipped"
fi

# Medical imaging (for some tracking modules)
if python -c "import monai" 2>/dev/null; then
    echo "  ✅ monai already installed, skipping"
else
    pip install monai==1.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ monai install skipped"
fi

# Computer vision deep learning
if python -c "import kornia" 2>/dev/null; then
    echo "  ✅ kornia already installed, skipping"
else
    pip install kornia -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ kornia install skipped"
fi

# Optional pose estimation frameworks (may not be needed but prevent warnings)
if python -c "import openpifpaf" 2>/dev/null; then
    echo "  ✅ openpifpaf already available, skipping"
else
    echo "  ⚠️ openpifpaf not installed (install manually if needed: pip install openpifpaf)"
fi

# OpenMMLab packages (heavyweight, only install if specifically needed)
if python -c "import mim" 2>/dev/null; then
    echo "  ✅ mim already available, skipping"
else
    echo "  ⚠️ mim not installed (install manually if needed: pip install openmim)"
fi

# Fix pkg_resources deprecation warning by pinning setuptools
echo "🔧 Fixing setuptools to avoid pkg_resources warnings..."
pip install "setuptools<81" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || echo "  ⚠️ setuptools downgrade skipped"

# Fix albumentations compatibility
echo "🔄 Fixing albumentations compatibility..."
# More comprehensive fix for is_check_shapes parameter
PRTREID_PATHS=$(find /opt/conda /usr/local .venv -name "transforms.py" -path "*/prtreid/data/transforms.py" 2>/dev/null)
if [ -n "$PRTREID_PATHS" ]; then
    for TRANSFORMS_PATH in $PRTREID_PATHS; do
        if [ -f "$TRANSFORMS_PATH" ]; then
            echo "  🔧 Fixing transforms at: $TRANSFORMS_PATH"
            # Create backup
            cp "$TRANSFORMS_PATH" "${TRANSFORMS_PATH}.backup" 2>/dev/null || true
            
            # Remove all variations of is_check_shapes parameter
            sed -i 's/is_check_shapes=False,\s*//g' "$TRANSFORMS_PATH"
            sed -i 's/is_check_shapes=True,\s*//g' "$TRANSFORMS_PATH"
            sed -i 's/,\s*is_check_shapes=False//g' "$TRANSFORMS_PATH"
            sed -i 's/,\s*is_check_shapes=True//g' "$TRANSFORMS_PATH"
            sed -i 's/is_check_shapes=[^,)]*,//g' "$TRANSFORMS_PATH"
            sed -i 's/,\s*is_check_shapes=[^,)]*)//g' "$TRANSFORMS_PATH"
            
            echo "  ✅ Fixed albumentations compatibility"
        fi
    done
else
    echo "  ⚠️ prtreid transforms.py not found yet - will fix on first import error"
fi

# Fix torchreid imports
echo "🔧 Fixing imports..."
find . -name "nn_matching.py" -path "*/sort/*" -exec sed -i 's/import torchreid/import prtreid as torchreid/g; s/axis=/dim=/g' {} \; 2>/dev/null || true

# Quick verification
echo "🧪 Verifying setup..."
python -c "
try:
    import torch, torchvision, prtreid, torchreid, albumentations
    print('✅ Core modules imported successfully')
    print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except Exception as e:
    print(f'⚠️ {e}')
    if 'is_check_shapes' in str(e):
        print('  🔧 Attempting runtime fix for albumentations...')
        import subprocess, glob
        transforms_files = glob.glob('/opt/conda/**/prtreid/data/transforms.py', recursive=True)
        transforms_files.extend(glob.glob('/usr/local/**/prtreid/data/transforms.py', recursive=True))
        transforms_files.extend(glob.glob('.venv/**/prtreid/data/transforms.py', recursive=True))
        for tf in transforms_files:
            subprocess.run(['sed', '-i', 's/is_check_shapes=[^,)]*,//g', tf], capture_output=True)
            subprocess.run(['sed', '-i', 's/,\s*is_check_shapes=[^,)]*//g', tf], capture_output=True)
        print('  ✅ Runtime fix applied')
"

# Create a helper script for manual fixing
cat > /workspaces/tracklab/fix_albumentations.sh << 'EOF'
#!/bin/bash
echo "🔧 Manual albumentations fix script"
find /opt/conda /usr/local .venv -name "transforms.py" -path "*/prtreid/data/transforms.py" 2>/dev/null | while read -r file; do
    if [ -f "$file" ]; then
        echo "Fixing: $file"
        cp "$file" "${file}.backup"
        sed -i 's/is_check_shapes=False,\s*//g' "$file"
        sed -i 's/is_check_shapes=True,\s*//g' "$file"
        sed -i 's/,\s*is_check_shapes=False//g' "$file"
        sed -i 's/,\s*is_check_shapes=True//g' "$file"
        sed -i 's/is_check_shapes=[^,)]*,//g' "$file"
        sed -i 's/,\s*is_check_shapes=[^,)]*//g' "$file"
        echo "Fixed: $file"
    fi
done
EOF
chmod +x /workspaces/tracklab/fix_albumentations.sh

echo "✅ Setup complete! Run: tracklab run -cn=./tracklab/configs/gamestate.yaml"
echo "💡 If you get 'is_check_shapes' error, run: ./fix_albumentations.sh"
echo "📦 PyTorch 2.0.1 with CUDA 11.8 installed"