#!/bin/bash
rm -Rf /root/.cache/fenics

./mesh.sh 0.006
mpirun -n 64 ./s1.sh
mv fecoda.log results/cg-its/64-44k.log

./mesh.sh 0.003
mpirun -n 64 ./s1.sh
mv fecoda.log results/cg-its/64-257k.log

./mesh.sh 0.002
mpirun -n 64 ./s1.sh
mv fecoda.log results/cg-its/64-790k.log
