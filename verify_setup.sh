#!/bin/bash
# TrackLab Dependency Verification Script
# Run this after container rebuilds to verify all dependencies are working

echo "🔍 TrackLab Dependency Verification"
echo "=================================="

# Check PyTorch
echo "Checking PyTorch..."
if python -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null; then
    echo "PyTorch OK"
else
    echo "❌ PyTorch FAILED"
fi

# Check MONAI
echo "Checking MONAI..."
if python -c "import monai; print('✅ monai:', monai.__version__)" 2>/dev/null; then
    echo "MONAI OK"
else
    echo "❌ MONAI FAILED"
fi

# Check Transformers
echo "Checking Transformers..."
if python -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null; then
    echo "Transformers OK"
else
    echo "❌ Transformers FAILED"
fi

# Check MMCV
echo "Checking MMCV..."
if python -c "import mmcv; print('✅ mmcv:', mmcv.__version__)" 2>/dev/null; then
    echo "MMCV OK"
else
    echo "❌ MMCV FAILED"
fi

# Check MMCV _ext module
echo "Checking MMCV _ext module..."
if python -c "import mmcv._ext; print('✅ mmcv._ext: OK')" 2>/dev/null; then
    echo "MMCV _ext OK"
else
    echo "❌ MMCV _ext FAILED"
fi

# Check MMOCR
echo "Checking MMOCR..."
if python -c "import mmocr; print('✅ mmocr:', mmocr.__version__)" 2>/dev/null; then
    echo "MMOCR OK"
else
    echo "❌ MMOCR FAILED"
fi

# Test TrackLab imports
echo "Testing TrackLab imports..."
if python -c "from tracklab.pipeline.jersey.mmocr_api import MMOCR; from tracklab.pipeline.reid.prtreid_api import PRTReId; print('✅ All TrackLab imports successful!')" 2>/dev/null; then
    echo "🎉 All TrackLab modules working correctly!"
    exit 0
else
    echo "❌ TrackLab imports FAILED!"
    echo ""
    echo "🔧 To fix this, try running:"
    echo "pip install monai==1.0.1 transformers==4.22.2 --force-reinstall"
    echo "pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --force-reinstall"
    echo "pip install 'mmdet>=3.0.0rc0,<3.1.0' mmocr==1.0.1 --force-reinstall"
    exit 1
fi