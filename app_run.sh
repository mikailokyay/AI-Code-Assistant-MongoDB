#!/bin/bash

echo "Creating repository directory: code-repo"
mkdir -p code-repo

cd code-repo

echo "Cloning the Ultralytics source code (This may take a moment)"
if [ ! -d "ultralytics" ]; then
    git clone https://github.com/ultralytics/ultralytics.git
else
    echo "Ultralytics repository already cloned. Skipping git clone."
fi

cd ..

echo "Starting data indexing"
python index_code.py

echo "Data indexing complete. Starting application"
python app.py
