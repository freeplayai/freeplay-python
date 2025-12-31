#!/usr/bin/env bash

echo "Setting up uv environment..."
echo

if ! which uv > /dev/null 2>&1; then
  echo "No uv found. Please install uv first:"
  echo "brew install uv"
  exit 1
fi

echo "Installing dependencies with uv..."
uv sync --group dev

echo
echo "UV Environment Info"
echo "=================="
echo "uv version: $(uv --version)"
echo "Python version: $(uv run python --version)"
echo "Virtual environment: $(uv venv --help > /dev/null 2>&1 && echo 'Managed by uv' || echo 'Not found')"
echo
