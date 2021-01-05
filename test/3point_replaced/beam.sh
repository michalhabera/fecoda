nprocs=$1

clscale=0.08
layers=60

fc=28
c=300
wc=0.6
ac=6.4

cd mesh/
gmsh -3 -clscale $clscale rebar.geo -setnumber layers $layers -setnumber length 8.0
python3 ../../convert.py --infile rebar.msh --outfile rebar

cd ../
mpirun -n $nprocs python3 -u beam.py --mesh rebar --force 2500 --fc $fc --c $c --wc $wc --ac $ac --disp_solver lu --ct SL
