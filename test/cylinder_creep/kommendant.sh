nprocs=$1

cd mesh/
gmsh -3 -clscale 0.5 cyl.geo -setnumber H 406.4 -setnumber D 152.4
python3 ../../convert.py --infile cyl.msh --outfile c152x406_5e-1
cd ../

rm -Rf /root/.cache/fenics/
mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 --sigma 14.48 \
    --tp 28 --ct SL --steps 35 --end 800 --out 01 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 --sigma 14.48 \
    --tp 90 --ct SL --steps 35 --end 800 --out 02 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 --sigma 16.55 \
    --tp 270 --ct SL --steps 35 --end 800 --out 05 -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py --files c152x406_5e-1_01.log c152x406_5e-1_02.log c152x406_5e-1_05.log \
    --out estim_01_02_05 --data c_054 --legend "\$t' = 28\$" "\$t' = 90\$" "\$t' = 270\$"


rm -Rf /root/.cache/fenics/
mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 \
    --q1 14 --q2 60 --q3 16 --q4 6 --sigma 14.48 --tp 28 --ct SL --steps 35 --end 800 --out 01 \
    -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 \
    --q1 14 --q2 60 --q3 16 --q4 6 --sigma 14.48 --tp 90 --ct SL --steps 35 --end 800 --out 02 \
    -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

mpirun -n $nprocs python3 -u cyl.py --mesh c152x406_5e-1 --fc 45 --c 419 --wc 0.38 --ac 4.34 \
    --q1 14 --q2 60 --q3 16 --q4 6 --sigma 16.55 --tp 270 --ct SL --steps 35 --end 800 --out 05 \
    -dispksp_type "preonly" -disppc_type "lu" -disppc_factor_mat_solver_type "mumps"

python3 plotting.py --files c152x406_5e-1_01.log c152x406_5e-1_02.log c152x406_5e-1_05.log \
    --out optim_01_02_05 --data c_054 --legend "\$t' = 28\$" "\$t' = 90\$" "\$t' = 270\$"

