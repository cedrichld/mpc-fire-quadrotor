#!/bin/bash

# Output file
output_file="prompt.txt"

# Clear the file if it already exists
> "$output_file"

# Check if at least one file is specified
if [ "$#" -eq 0 ]; then
    echo "Please specify the Python files to include."
    exit 1
fi

# Loop through the specified files
for file in "$@"; do
    if [[ -f "$file" && "$file" == *.py || *.m ]]; then
        # Print the filename
        echo "FILENAME: $file" >> "$output_file"
        
        # Print the file contents
        echo -e "\nCONTENTS:\n\n" >> "$output_file"
        cat "$file" >> "$output_file"
        
        # Add a separator for readability
        echo -e "\n\n\n\n=========================\n\n=========================\n\n=========================\n\n" >> "$output_file"
    else
        echo "Skipping invalid or non-Python file: $file"
    fi
done

echo "Specified Python/Matlab files and their contents have been written to $output_file."
