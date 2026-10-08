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
# Force CMake to accept the older configuration by passing the policy flag
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
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

# SINGER binaries are pre-compiled in the repository's releases folder
RELEASE_DIR="$SRC_DIR/SINGER/releases/singer-0.1.9-beta-linux-x86_64"

echo "Symlinking SINGER binaries from releases to $BIN_DIR..."
if [ -d "$RELEASE_DIR" ]; then
    # Force executable permissions on all files in the release directory
    chmod +x "$RELEASE_DIR"/*

    for exe in "$RELEASE_DIR"/*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            ln -sf "$exe" "$BIN_DIR/"
        fi
    done
    echo "SINGER successfully installed."
else
    echo "Error: SINGER release directory not found at $RELEASE_DIR"
fi


# --- 4. Update ~/.bashrc ---
BASHRC="$HOME/.bashrc"

echo ""
echo "[4/4] Configuring PATH..."

# Check if BIN_DIR is already in the bashrc file
if grep -qF "$BIN_DIR" "$BASHRC"; then
    echo "$BIN_DIR is already in your $BASHRC."
else
    echo "Appending $BIN_DIR to PATH in $BASHRC..."
    echo "" >> "$BASHRC"
    echo "# Added by popgen tools install script" >> "$BASHRC"
    echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$BASHRC"
    
    source ~/.baschrc
fi
