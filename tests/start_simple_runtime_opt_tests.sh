#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../src

python_cmd=python3
command -v $python_cmd >/dev/null 2>&1 || python_cmd=python

clean=0

test_and_clean () {
    test $? -eq 0 || { echo "failed" ; exit 1; }

    if [ $clean -eq 1 ]; then
        rm *.pdf
        rm *.png
        rm *.stat
        rm *.csv
    fi
}

sc_q_order=2
sc_p_order=2
mc_num_samples=1000


# Smoke test
$python_cmd simple_runtime_opt_solver_tests.py --smoketest
test_and_clean

# Linear Solver - SC
$python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all"
test_and_clean

 Linear Solver - MC
$python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all"
test_and_clean

# Parallel Solver - SC
$python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --parallel
test_and_clean

# Parallel Solver - MC
$python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --parallel
test_and_clean

# MpiSolverSolver - DWP - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi \
                                    --opt_strategy "DYNAMIC" --opt_algorithm "FCFS"
test_and_clean

# MpiSolverSolver - DWP_OPT - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi \
                                    --opt_strategy "DYNAMIC" --opt_algorithm "LPT" --opt_runtime
test_and_clean

# MpiSolverSolver - DWP - MC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi \
                                    --opt_strategy "DYNAMIC" --opt_algorithm "FCFS"
test_and_clean

# MpiSolver - SWP - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "FCFS"
test_and_clean

# MpiSolver - SWP_OPT - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "MULTIFIT"
test_and_clean

# MpiSolver - SWP - MC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "FCFS"
test_and_clean

# MpiSolver - SWPT - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" --mpi_combined_parallel \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "FCFS"
test_and_clean

# MpiSolver - SWPT_OPT - SC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "sc" --sc_q_order $sc_q_order --sc_p_order $sc_p_order \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" --mpi_combined_parallel \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "MULTIFIT" --opt_runtime
test_and_clean

# MpiSolver - SWPT - MC
mpiexec -n 4 $python_cmd simple_runtime_opt_solver_tests.py \
                                    --model "runtime" \
                                    --uq_method "mc" --mc_numevaluations $mc_num_samples \
                                    --uncertain "all" \
                                    --mpi --mpi_method "MpiSolver" --mpi_combined_parallel \
                                    --opt_strategy "FIXED_LINEAR" --opt_algorithm "FCFS"
test_and_clean
