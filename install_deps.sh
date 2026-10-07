#!/bin/bash

# Exit immediately if any command fails
set -e

# Define local installation paths (no sudo required)
PREFIX="$HOME/.local"
BIN_DIR="$PREFIX/bin"
SRC_DIR="$PREFIX/src"

# Ensure the directories exist
mkdir -p "$BIN_DIR"
mkdir -p "$SRC_DIR"

echo "====================================================="
echo " Preparing to install SLiM, Relate, and SINGER"
echo " Installation Prefix: $PREFIX"
echo " Source Directory:    $SRC_DIR"
echo " Bin Directory:       $BIN_DIR"
echo "====================================================="

# --- 1. Install SLiM (MesserLab) ---
echo ""
echo "[1/3] Installing SLiM..."
cd "$SRC_DIR"

if [ ! -d "SLiM" ]; then
    git clone https://github.com/MesserLab/SLiM.git
else
    echo "SLiM directory already exists. Pulling latest..."
    cd SLiM && git pull && cd ..
fi

cd SLiM
mkdir -p build
cd build

# Configure CMake to install into ~/.local instead of /usr/local
cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
make -j4
make install

echo "SLiM successfully installed to $BIN_DIR"


# --- 2. Install Relate (MyersGroup) ---
echo ""
echo "[2/3] Installing Relate..."
cd "$SRC_DIR"

if [ ! -d "relate" ]; then
    git clone https://github.com/MyersGroup/relate.git
else
    echo "Relate directory already exists. Pulling latest..."
    cd relate && git pull && cd ..
fi

cd relate
mkdir -p build
cd build

# Relate compiles binaries directly into the relate/bin folder
cmake ..
make -j4

# Create symlinks for all compiled binaries so they are accessible in the PATH
echo "Symlinking Relate binaries to $BIN_DIR..."
for exe in "$SRC_DIR"/relate/bin/*; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        ln -sf "$exe" "$BIN_DIR/"
    fi
done

echo "Relate successfully installed."


# --- 3. Install SINGER (popgenmethods) ---
echo ""
echo "[3/3] Installing SINGER..."
cd "$SRC_DIR"

if [ ! -d "SINGER" ]; then
    git clone https://github.com/popgenmethods/SINGER.git
else
    echo "SINGER directory already exists. Pulling latest..."
    cd SINGER && git pull && cd ..
fi

cd SINGER
mkdir -p build
cd build

# SINGER uses a standard CMake configuration
cmake ..
make -j4

# SINGER builds the 'singer_master' and 'convert_to_tskit' binaries.
# We symlink any compiled executables from the build directory.
echo "Symlinking SINGER binaries to $BIN_DIR..."
for exe in "$SRC_DIR"/SINGER/build/*; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        ln -sf "$exe" "$BIN_DIR/"
    fi
done

echo "SINGER successfully installed."


# --- 4. Final PATH configuration check ---
echo ""
echo "====================================================="
echo " Installation Complete!"
echo "====================================================="

# Check if ~/.local/bin is actually in the user's PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "WARNING: $BIN_DIR is not currently in your system PATH."
    echo ""
    echo "To use 'slim', 'Relate', and 'singer_master' from anywhere,"
    echo "add the following line to your ~/.bashrc or ~/.zshrc file:"
    echo ""
    echo "    export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
    echo "Then reload your terminal by running: source ~/.bashrc"
else
    echo "$BIN_DIR is already in your PATH. You are ready to go!"
    echo "Try running 'slim -v', 'Relate', and 'singer_master' to verify."
fi
