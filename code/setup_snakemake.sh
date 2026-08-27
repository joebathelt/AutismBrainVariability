#!/bin/bash
# Setup script for Snakemake workflow

echo "Setting up Snakemake workflow for Brain Compensation project..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Install Snakemake in the BrainComp environment
echo "Installing Snakemake in the BrainComp environment..."
conda install -n BrainComp -c conda-forge -c bioconda snakemake -y

# Create necessary directories
echo "Creating required directories..."
mkdir -p ../data/results
mkdir -p logs

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml to specify your input data filenames"
echo "2. Activate the environment: conda activate BrainComp"
echo "3. Test the workflow: snakemake --dry-run --cores 1"
echo "4. Run the workflow: snakemake --cores 4 --use-conda"
echo ""
echo "For more details, see README_SNAKEMAKE.md"
