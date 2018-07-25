#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../src

python_cmd=python
#python_cmd=python2
#python_cmd=python3

# Linear Solver - SC
$python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order 2 --sc_p_order 2 \
                                    --uncertain "all"

# Linear Solver - MC
#$python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all"



# Parallel Solver - SC
#$python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                    --uncertain "all" \
#                                    --parallel

# Parallel Solver - MC
#$python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all" \
#                                    --parallel


# MpiPoolSolver - SC
#mpiexec -n 4 $python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                    --uncertain "all" \
#                                    --mpi

# MpiPoolSolver - MC
#mpiexec -n 4 $python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all" \
#                                    --mpi

# MpiSolverOld - SC
#mpiexec -n 4 $python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                    --uncertain "all" \
#                                    --mpi --mpi_method "old"

# MpiSolverOld - MC
#mpiexec -n 4 $python_cmd simple_example_testmodel.py \
#                                    --model "testmodel" \
#                                    --uq_method "mc" --mc_numevaluations 1000 \
#                                    --uncertain "all" \
#                                    --mpi --mpi_method "old"
