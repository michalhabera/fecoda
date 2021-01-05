#!/bin/bash
python3 -u beam.py --mesh beam --fc 29.0 --c 279 --wc 0.6 --ac 7.05 --ct SL \
    --q1 30 --ft 3.1 --Gf 30 --steps 150 --displ 0.000004 --out malvar-s1 \
    -dispksp_type "bcgs" -disppc_type "gamg" -disppc_gamg_coarse_eq_limit "1000" \
    -dispmg_levels_ksp_type "chebyshev" -dispmg_levels_pc_type "sor" \
    -disppc_factor_mat_solver_type "mumps" -dispksp_rtol "1.0e-12" \
    -dispksp_max_it "200"
