#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "=== 1. Detecting CUDA Version ==="
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc could not be found. Please ensure CUDA is loaded."
    exit 1
fi

# Extract the release version (e.g., "12.4" or "11.8")
CUDA_RAW=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Detected raw CUDA Version: $CUDA_RAW"

# Map to the closest available PyTorch wheel index
if [[ "$CUDA_RAW" == "12.6"* ]] || [[ "$CUDA_RAW" == "12.7"* ]]; then
    TORCH_CUDA="cu126"
elif [[ "$CUDA_RAW" == "12.4"* ]] || [[ "$CUDA_RAW" == "12.5"* ]]; then
    TORCH_CUDA="cu124"
elif [[ "$CUDA_RAW" == "12.1"* ]] || [[ "$CUDA_RAW" == "12.2"* ]] || [[ "$CUDA_RAW" == "12.3"* ]]; then
    TORCH_CUDA="cu121"
elif [[ "$CUDA_RAW" == "11.8"* ]] || [[ "$CUDA_RAW" == "11.9"* ]]; then
    TORCH_CUDA="cu118"
else
    # Fallback: remove the dot (e.g. 13.0 -> 130)
    TORCH_CUDA="cu$(echo $CUDA_RAW | cut -d. -f1,2 | tr -d '.')"
fi

echo "Mapping to PyTorch wheel index: $TORCH_CUDA"

echo "=== 2. Installing PyTorch ==="
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "=== 3. Installing PyTorch Geometric from Source ==="
# Ninja is required for faster C++ compilation and is utilized by --no-build-isolation
pip install ninja setuptools wheel

# Install core library
pip install torch_geometric

export MAX_JOBS=$(nproc)

# Install extensions from source without build isolation
# (This forces pip to compile using the torch version installed in Step 2)
FORCE_CUDA=1 pip install --verbose --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git
FORCE_CUDA=1 pip install --verbose --no-build-isolation torch_scatter torch_sparse torch_cluster torch_spline_conv

echo "=== 4. Running External Simulators Setup ==="
if [ -f "install_deps.sh" ]; then
    chmod +x install_deps.sh
    ./install_deps.sh
else
    echo "Warning: install_deps.sh not found in the current directory."
fi

echo "=== 5. Installing Package and Requirements ==="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install the package itself (using pip install -e . is preferred over python setup.py install for modern builds)
pip install -e .

echo "=== Setup Complete! ==="