#!/bin/bash

VER=$1

if [[ -z $VER ]]; then
  echo "You need to provide a version string."
  exit 1
fi

echo $VER
rm -rf build/ dist/

echo "__version__ = '${VER}'" > nmtpytorch/__init__.py
#git commit nmtpytorch/__init.py
python setup.py sdist bdist_wheel

# Upload to TestPyPI
#twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
twine upload dist/*
