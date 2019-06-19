#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/data/repos/repos_tum/chaospy_run.git
export PYTHONPATH=$PYTHONPATH:../src

python_cmd=python3
command -v $python_cmd >/dev/null 2>&1 || python_cmd=python

check () {
    test $? -eq 0 || { echo "failed" ; exit 1; }
}

sc_q_order=2
sc_p_order=2
mc_num_samples=1000

# Smoke test
$python_cmd simple_example_testmodel.py --smoketest
check

# Linear Solver - SC
$python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all"
check

# Linear Solver - MC
$python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all"
check

# Parallel Solver - SC
$python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --parallel
check

# Parallel Solver - MC
$python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --parallel
check

# MpiPoolSolver - SC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi
check

# MpiPoolSolver - MC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi
check

# MpiSolverOld - SWP - SC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "old"
check

# MpiSolverOld - SWP - MC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi --mpi_method "old"
check

# MpiSolverOld - SWPT - SC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "old" --mpi_combined_parallel
check

# MpiSolverOld - SWPT - MC
mpiexec -n 4 $python_cmd simple_example_testmodel.py \
                                    --model "testmodel" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi --mpi_method "old" --mpi_combined_parallel
check