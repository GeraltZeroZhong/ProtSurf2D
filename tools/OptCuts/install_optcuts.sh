#!/bin/bash
# Install OptCuts_bin into the current Conda environment

echo "Installing OptCuts into Conda..."

cp "$(dirname "$0")/OptCuts_bin" "$CONDA_PREFIX/bin/OptCuts_bin"
chmod +x "$CONDA_PREFIX/bin/OptCuts_bin"

echo "Installation complete!"
echo "You can now run OptCuts_bin directly in your Conda environment."
echo "Example: OptCuts_bin 10 input.obj 0.999 1 0 4.1 1 0 mytest"