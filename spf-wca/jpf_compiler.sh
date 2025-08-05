#!/bin/bash

# Directory containing the .jpf files
JPF_DIR="/Users/dankoh/jpf/spf-wca/src/examples/own"

# Iterate over each .jpf file in the directory
for jpf_file in "$JPF_DIR"/*.jpf; 
do
  echo "Running JPF for $jpf_file"
  ../jpf-core/bin/jpf "$jpf_file"
done