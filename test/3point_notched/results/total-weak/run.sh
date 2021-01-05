#!/bin/bash
rm -Rf /root/.cache/fenics

./mesh.sh 0.015
mpirun -n 1 ./s1.sh
mv fecoda.log results/total-weak/1.log

./mesh.sh 0.006
mpirun -n 8 ./s1.sh
mv fecoda.log results/total-weak/8.log

./mesh.sh 0.0046
mpirun -n 16 ./s1.sh
mv fecoda.log results/total-weak/16.log

./mesh.sh 0.0035
mpirun -n 32 ./s1.sh
mv fecoda.log results/total-weak/32.log

./mesh.sh 0.0027
mpirun -n 64 ./s1.sh
mv fecoda.log results/total-weak/64.log

# ./mesh.sh 0.0021
# mpirun -n 128 ./s1.sh
# mv fecoda.log results/total-weak/128.log
