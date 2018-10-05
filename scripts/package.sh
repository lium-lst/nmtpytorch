#!/bin/bash

VER=$1

if [[ -z $VER ]]; then
  echo "You need to provide a version string."
  exit 1
fi

rm -rf build/ dist/

echo "Preparing $VER"
echo "__version__ = '${VER}'" > nmtpytorch/__init__.py

git commit nmtpytorch/__init__.py -m "bump version to ${VER}"
git push origin master
git tag -a "v${VER}" -m "Version ${VER}"
git push origin --tags

# prep packages
python setup.py sdist bdist_wheel

#twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # Upload to TestPyPI
twine upload dist/*  # Upload to PyPI
