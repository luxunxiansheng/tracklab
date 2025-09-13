# TrackLab DevContainer Configuration

## Overview
This devcontainer is configured for TrackLab development with Python 3.9 and UV package manager.

## Key Features

- **Python 3.9.23**: Compatible with TrackLab requirements
- **UV Package Manager**: Fast dependency resolution with Chinese mirrors
- **GPU Support**: CUDA-enabled with `--gpus=all`
- **Chinese Mirror**: Uses Tsinghua University PyPI mirror for faster downloads
- **Auto-activation**: Virtual environment automatically activated in terminal
- **Tested Dependencies**: Uses proven working versions (PyTorch 1.13.1, Lightning 2.0.9)


## Environment Setup
The container automatically:
1. Creates Python 3.9 virtual environment using UV
2. Installs all TrackLab dependencies from `pyproject.toml`
3. Installs compatible PyTorch >= 2.1
4. Configures git user settings
5. Activates environment on startup

## Chinese Mirror Configuration
- Primary index: `https://pypi.tuna.tsinghua.edu.cn/simple`
- Fallback index: `https://pypi.org/simple/`
- Extended timeout: 300 seconds for large packages

## VS Code Integration
- Python interpreter: `/workspaces/tracklab/.venv/bin/python`
- Pylance type checking enabled
- Auto-import completions
- Jupyter notebook support

## Manual Commands
If you need to manually manage the environment:

```bash
# Activate environment
source /workspaces/tracklab/.venv/bin/activate

# Install additional packages
uv pip install package_name --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Test TrackLab
python -c "import tracklab; print('✅ Working!')"
python main.py --help
```

## Troubleshooting

- If packages fail to install, increase `UV_HTTP_TIMEOUT`
- For dependency conflicts, try `uv sync --refresh`
- Check Python version with `python --version` (should be 3.9.23)
- Expected working versions: PyTorch 1.13.1, Lightning 2.0.9, TorchVision 0.14.1