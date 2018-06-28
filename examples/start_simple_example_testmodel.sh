#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../src

# Linear Solver - SC
python simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
                                    --uncertain "all"

# Linear Solver - MC
#python simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all"



# Parallel Solver - SC
#python simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                    --uncertain "all" \
#                                    --parallel

# Parallel Solver - MC
#python simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all" \
#                                    --parallel


# MpiPoolSolver - SC
#mpiexec -n 4 python simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                    --uncertain "all" \
#                                    --mpi

# MpiPoolSolver - MC
#mpiexec -n 4 python simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all" \
#                                    --mpi
