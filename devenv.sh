#!/usr/bin/env bash

echo "Checking Python version..."
echo

if ! which pyenv > /dev/null 2>&1; then
  echo "No pyenv found. Please install."
fi

if pyenv versions | grep '^[*]*\s*3.8.*' > /dev/null 2>&1; then
  echo "Python 3.8 is installed."
else
  echo "Installing Python 3.8..."
  pyenv install 3.8
fi

poetry env use 3.8

echo
echo "Poetry Environment"
echo "=================="
poetry env info
echo
