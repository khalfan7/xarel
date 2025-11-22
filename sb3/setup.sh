#!/bin/bash
# Setup script for SB3 SurRoL training

set -e  # Exit on error

echo "======================================"
echo "Setting up SB3 for SurRoL training"
echo "======================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "Please activate your conda environment first:"
    echo "  conda activate arel"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Optional: Install WandB for better logging
echo ""
read -p "Install WandB for advanced logging? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install wandb
    echo "Run 'wandb login' to connect your account"
fi

# Make scripts executable
echo ""
echo "🔧 Making scripts executable..."
chmod +x hpc/*.sh
chmod +x hpc/*.sbatch

# Create logs directory if it doesn't exist
echo ""
echo "📁 Creating output directories..."
mkdir -p ../logs
mkdir -p exp_sb3
mkdir -p exp_sb3/tensorboard
mkdir -p exp_sb3/checkpoints

# Verify SurRoL is accessible
echo ""
echo "🔍 Verifying SurRoL installation..."
if [ -d "../SurRoL/surrol" ]; then
    echo "✅ SurRoL directory found"
else
    echo "❌ SurRoL directory not found at ../SurRoL/"
    echo "Please ensure SurRoL is installed in the parent directory"
fi

echo ""
echo "======================================"
echo "✅ Installation complete!"
echo "======================================"
echo ""
echo "Quick Start:"
echo "  cd $SCRIPT_DIR"
echo "  python train_sb3.py"
echo ""
echo "HPC Training:"
echo "  cd hpc"
echo "  sbatch train_sb3.sbatch"
echo ""
echo "Monitor Training:"
echo "  tensorboard --logdir exp_sb3"
echo ""
echo "Documentation:"
echo "  cat README.md"
echo "  cat docs/TRAINING_GUIDE.md"
echo ""
