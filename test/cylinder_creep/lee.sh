nprocs=$1

cd mesh/
gmsh -3 -clscale 0.5 cyl.geo -setnumber H 300 -setnumber D 150
python3 ../../convert.py --infile cyl.msh --outfile c150x300_5e-1
cd ../

# wc = 0.5
rm -Rf /root/.cache/fenics/
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 35 --c 350 --wc 0.5 --ac 5.09 \
    --sigma 2.85 --tp 1.0 --end 30 --steps 20 --out 25 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 35 --c 350 --wc 0.5 --ac 5.09 \
    --sigma 5.24 --tp 3.0 --end 30 --steps 20 --out 26 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 35 --c 350 --wc 0.5 --ac 5.09 \
    --sigma 10.55 --tp 28.0 --end 30 --steps 20 --out 28 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py --files c150x300_5e-1_26.log c150x300_5e-1_25.log c150x300_5e-1_28.log --out 26_25_28 --data d_024 --legend "\$t' = 1\$" "\$t' = 3\$" "\$t' = 28\$"


# wc = 0.3
rm -Rf /root/.cache/fenics/
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 57 --c 583 --wc 0.3 --ac 2.73 \
    --sigma 7.07 --tp 1.0 --end 30 --steps 20 --out 17 --io -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 57 --c 583 --wc 0.3 --ac 2.73 \
    --sigma 9.35 --tp 3.0 --end 30 --steps 20 --out 18 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py --files c150x300_5e-1_17.log c150x300_5e-1_18.log --out 17_18 --data d_024 --legend "\$t' = 1\$" "\$t' = 3\$" "\$t' = 28\$"


# wc = 0.6
rm -Rf /root/.cache/fenics/
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 26 --c 292 --wc 0.6 --ac 6.43 \
    --sigma 1.56 --tp 1.0 --out 29 --end 30 --steps 20 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 26 --c 292 --wc 0.6 --ac 6.43 \
    --sigma 3.44 --tp 3.0 --out 30 --end 30 --steps 20 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"
mpirun -n $nprocs python3 -u cyl.py --mesh c150x300_5e-1 --fc 26 --c 292 --wc 0.6 --ac 6.43 \
    --sigma 7.94 --tp 28.0 --out 32 --end 30 --steps 20 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py --files c150x300_5e-1_29.log c150x300_5e-1_30.log c150x300_5e-1_32.log --out 29_30_32 --data d_024 --legend "\$t' = 1\$" "\$t' = 3\$" "\$t' = 28\$"
