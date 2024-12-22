#!/bin/bash

# Get the directory of the project root (parent of the scripts folder)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Export PYTHONPATH using the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Green-colored message for confirmation
echo -e "\n\033[1;32mPYTHONPATH set to:\033[0m $PYTHONPATH"

# Friendly setup instructions
echo -e "\n\033[1;34mImportant:\033[0m It is recommended to run this project in a Python virtual environment (for pydrake)."
echo "If you do not have a virtual environment set up, consider creating one in this repo."

echo -e "\nTo create a virtual environment and install dependencies, run: \033[1;36m./venv.sh\033[0m"
echo -e "This will download pydrake, scipy, and all other required dependencies for the project."

echo -e "\nOnce the virtual environment is set up, activate it by running: \033[1;36msource env/bin/activate\033[0m"
