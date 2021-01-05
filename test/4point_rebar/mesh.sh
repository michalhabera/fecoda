cd mesh/
gmsh -3 -clscale $1 rebar_b1.geo -setnumber layers $2
python3 ../../convert.py --infile rebar_b1.msh --outfile rebar_b1a_$1
cd ../