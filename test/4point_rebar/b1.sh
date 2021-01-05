#!/bin/bash
fc=24.8
c=350
wc=0.6
ac=3.0

python3 -u beam.py --mesh rebar_b1a_$1 --force 11.8 --fc $fc --c $c --wc $wc --ac $ac --Gf 50.0 --out b1 \
    -dispksp_type "bcgs" -disppc_type "gamg" -disppc_gamg_coarse_eq_limit "1000" \
    -dispmg_levels_ksp_type "chebyshev" -dispmg_levels_pc_type "sor" \
    -disppc_factor_mat_solver_type "mumps" -dispksp_rtol "1.0e-12" \
    -dispksp_max_it "200" -mat_mumps_icntl_24 "1"
