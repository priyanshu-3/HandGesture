#!/bin/bash
# Quick start script for Hand Gesture 3D Interaction

echo "=========================================="
echo "Hand Gesture 3D Interaction System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting application..."
echo "=========================================="
echo ""

# Run the application
python main.py

# Deactivate when done
deactivate

