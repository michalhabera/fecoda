#!/bin/bash
python3 -u beam.py --mesh beam --fc 58.9 --c 613 --wc 0.4 --ac 2.4 --ct SL \
    --q1 30 --ft 4.2 --Gf 30 --steps 150 --displ 0.000004 --out malvar-s2 \
    -dispksp_type "bcgs" -disppc_type "gamg" -disppc_gamg_coarse_eq_limit "1000" \
    -dispmg_levels_ksp_type "chebyshev" -dispmg_levels_pc_type "sor" \
    -disppc_factor_mat_solver_type "mumps" -dispksp_rtol "1.0e-12" \
    -dispksp_max_it "200"
