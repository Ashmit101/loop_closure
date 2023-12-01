#!/bin/bash

git clone --recurse-submodules  https://github.com/Sologala/pyfbow.git
cd pyfbow

./build_flow.sh

python3 setup.py install
