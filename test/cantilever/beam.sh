nprocs=$1

fc=30
c=300
wc=0.6
ac=6.0

rm -Rf /root/.cache/fenics
mpirun -n $nprocs python3 -u beam.py --fc $fc --c $c --wc $wc --ac $ac --ct SL \
    -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py