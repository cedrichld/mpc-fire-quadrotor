#!/bin/bash
set -e  # Exit immediately if a command fails

# Get the project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Navigate to the project root directory
cd "$PROJECT_ROOT"

echo -e "033[1;32mThis script takes about 5 min to build the venv properly...\033[0m"

# Remove any existing virtual environment in the project root
if [ -d "env" ]; then
    echo -e "\033[1;33mRemoving existing virtual environment...\033[0m"
    rm -rf env
fi

echo -e "\033[1;34mCreating a new virtual environment in $PROJECT_ROOT/env...\033[0m"
python3 -m venv env

echo -e "\033[1;34mActivating the virtual environment...\033[0m"
source env/bin/activate

# Install required dependencies
echo -e "\033[1;32mInstalling pydrake...\033[0m"
pip install drake

# echo -e "\033[1;32mInstalling numpy...\033[0m"
# pip install numpy

echo -e "\033[1;32mInstalling matplotlib...\033[0m"
pip install matplotlib

echo -e "\033[1;32mInstalling scipy...\033[0m"
pip install scipy

echo -e "\033[1;32mInstalling tqdm...\033[0m"
pip install tqdm

echo -e "033[1;32mRelaunching env/bin/activate and setup.sh...\033[0m"
source env/bin/activate
source scripts/setup.sh

clear && echo -e "\033[1;32mVirtual environment setup complete!"
echo -e "\033[0m \nActivate it with: \033[1;36msource env/bin/activate\033[0m\n"