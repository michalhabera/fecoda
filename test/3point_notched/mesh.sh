#!/bin/bash
cd mesh/
gmsh -3 -clscale 1.0 beam.geo -setnumber scale $1
python3 ../../convert.py --infile beam.msh --outfile beam
cd ../
