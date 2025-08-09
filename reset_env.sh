#!/bin/bash

# Stop on first error
set -e

echo "Removing existing virtual environment..."
rm -rf .venv

echo "Creating new virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Environment setup complete."
