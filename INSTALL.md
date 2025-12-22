# Installation Guide

This guide provides instructions for installing Attend-and-Excite using pip and a virtual environment.

## Prerequisites

- Python 3.8, 3.9, or 3.10
- CUDA-capable GPU (recommended for running Stable Diffusion)
- Git (for cloning the repository)

## Installation Methods

### Method 1: Basic Installation (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yuval-alaluf/Attend-and-Excite.git
   cd Attend-and-Excite
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

   This will install all required dependencies automatically.

### Method 2: Installation with Development Tools

For development work with Jupyter notebooks:

```bash
pip install -e ".[dev]"
```

### Method 3: Installation with All Optional Dependencies

To install all optional dependencies (including metrics and taming transformers):

```bash
pip install -e ".[all]"
```

## Verifying Installation

To verify the installation, run:

```python
python -c "from attend_and_excite import AttendAndExcitePipeline; print('Installation successful!')"
```

## Usage After Installation

### Command Line

Once installed, you can run the script from anywhere:

```bash
cd Attend-and-Excite
python run.py --prompt "a cat and a dog" --seeds [0] --token_indices [2,5]
```

### In Python Scripts

```python
from attend_and_excite import AttendAndExcitePipeline, RunConfig
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = AttendAndExcitePipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to(device)
```

### In Jupyter Notebooks

After installing with `[dev]` option, launch Jupyter:

```bash
jupyter notebook notebooks/generate_images.ipynb
```

## Troubleshooting

### ImportError with diffusers

If you encounter import errors related to `diffusers`, ensure you have the correct versions:

```bash
# Reinstall the versions declared in pyproject.toml
pip install -e . --upgrade
```

Note: SD3/SD3.5 training (LoRA/DreamBooth) requires a newer Diffusers stack than legacy SD1/SD2 examples.

### CUDA/GPU Issues

Ensure you have the correct PyTorch version for your CUDA version. Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for specific installation instructions.

### Memory Issues

If you encounter memory issues, try using float16 precision:

```python
pipeline = AttendAndExcitePipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to(device)
```

## Updating

To update to the latest version:

```bash
cd Attend-and-Excite
git pull
pip install -e . --upgrade
```

## Uninstallation

To uninstall:

```bash
pip uninstall attend-and-excite
```

To remove the virtual environment:

```bash
deactivate
rm -rf venv/
```

## Legacy Installation (Conda)

If you prefer using the original conda environment:

```bash
conda env create -f environment/environment.yaml
conda activate ldm
```

Note: The conda method is now deprecated in favor of pip installation.
