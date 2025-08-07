#!/bin/bash

# --- Configuration ---
# Specify the name of the file you want to modify.
# This is likely the 'boundary' file in your polyMesh directory.
FILENAME="./constant/polyMesh/boundary"

# Check if the file exists before trying to edit it.
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' not found."
    exit 1
fi

echo "Editing file: $FILENAME"

# --- Operations ---
# Use sed to perform the edits in-place and create a backup file (.bak)
# - The first expression deletes the multi-line 'front' block.
# - The second expression finds a line containing only the number 6 and replaces it with 5.
sed -i.bak \
    -e '/^\s*front\s*$/,/^\s*}\s*$/d' \
    -e 's/^\s*6\s*$/5/' \
    "$FILENAME"

echo "Done! A backup of the original file has been saved as '$FILENAME.bak'."
