# UQEF - Uncertainty Quantification Execution Framework

A  Python framework for efficient uncertainty quantification (UQ) of computational models with support for custom models, multiple UQ methods, parallel computing, and statistical analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/) [![pypi](https://img.shields.io/pypi/v/uqef)](https://pypi.org/project/uqef)

## Overview

UQEF (Uncertainty Quantification Execution Framework) is designed to facilitate forward uncertainty quantification analyses for computational models. It provides a unified interface for various UQ methodologies, enabling researchers and engineers to quantify uncertainties in their models efficiently and effectively. With its various parallelisation methods, it supports prototyping and model execution on a single computer as well as productive runs on a cluster with the same code--just by setting some parameters. UQEF is mainly based on [Chaospy](https://github.com/jonathf/chaospy), which provides the implementation of the UQ methods.

The framework is particularly well-suited for:
- Scientific computing applications requiring uncertainty analysis (via the Monte Carlo simulations or polynomial chaos expansion-based methods)
- Sensitivity analysis of model parameters
- Parallel execution on single computers, up to clusters and High-performance computing environments with MPI support

## Key Features

### Multiple UQ Methods
- **Monte Carlo (MC)**: Random sampling-based uncertainty propagation
- **Stochastic Collocation (SC)**: Polynomial chaos expansion using collocation methods or pseudo-spectral method
- **Saltelli Method**: Global sensitivity analysis using Sobol indices based on MC samples
- **Ensemble Simulations**: User-defined parameter sets for ensemble runs

### Parallel Computing Support
- **Multiprocessing**: Shared memory parallelization for multi-core systems
- **MPI**: Distributed memory parallelization using `mpi4py`
- **Combined Parallelization**: Hybrid MPI + multiprocessing approaches
- **Runtime Optimization**: Intelligent task scheduling (LPT, SPT, MULTIFIT algorithms)
- Automatic runtime measurements, predictions, and optimised scheduling

### Statistical Analysis
- Comprehensive statistical measures (mean, variance, standard deviation, percentiles, ...)
- Generation of numerically constructed probability distributions as the quantity of interest.
- Sobol sensitivity indices (first-order, total-order, and higher-order)

### Flexible Configuration and Parametrisation
- JSON-based configuration files for parameter definitions
- Command-line interface with extensive options
- Custom model integration capabilities
- Custom statistics integration capabilities

### Visualization & Output
- Automatic generation of statistical plots in custom statistics
- Multiple output formats for data (pickle, CSV, NetCDF)

## Installation

### Prerequisites
- Python 3.x
- MPI implementation (e.g., OpenMPI, MPICH) for parallel execution

### Install from pypi
```bash
pip install uqef
```

### Install from source

```bash
git clone https://github.com/flo2k/UQEF.git
cd UQEF
pip install -e .
```

### Dependencies

UQEF requires the following Python packages ([requirements.txt](requirements.txt)):
- `chaospy` - For probabilistic modelling and forward uncertainty propagation (i.e., via the MC sampling methods or Polynomial chaos expansion-based methods)
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `mpi4py` - MPI support
- `joblib` - Parallel computing utilities
- `scikit-learn` - Machine learning (for regression)
- `dill` - Serialization
- `tabulate` - Table formatting
- `seaborn` - Statistical visualizations
- `more-itertools` - Advanced iteration tools

All dependencies will be automatically installed when using `pip install`.

## Quick Start

### Basic Example

```python
import matplotlib
matplotlib.use('Agg')
import chaospy as cp
import uqef

# Instantiate UQsim
uqsim = uqef.UQsim()

# Setup uncertain parameters
if uqsim.is_master():
    uqsim.setup_nodes(["uncertain_param_1", "uncertain_param_2"])
    
    # Define parameter distributions
    uqsim.simulationNodes.setDist("uncertain_param_1", cp.Normal(3, 0.1))
    uqsim.simulationNodes.setDist("uncertain_param_2", cp.Normal(6, 0.1))

# Setup and run simulation
uqsim.setup()
uqsim.simulate()

# Calculate and display statistics
uqsim.calc_statistics()
uqsim.print_statistics()
uqsim.plot_statistics()
uqsim.save_statistics()

# Clean up
uqsim.tear_down()
```

Run with:
```bash
python your_script.py
```
This starts your UQ simulation with the `testmodel`, which is the default model, until some other model (see: [Model settings](#model-settings) and [Custom model and statistics integration](#custom-model-and-statistics-integration)) is set.

### Using Configuration Files

Create a `config.json` file:

```json
{
  "parameters": [
    {
      "name": "uncertain_param_1",
      "distribution": "Normal",
      "mu": 3,
      "sigma": 0.1
    },
    {
      "name": "uncertain_param_2",
      "distribution": "Uniform",
      "lower": 2,
      "upper": 6
    }
  ]
}
```

Run with configuration:

```bash
python your_script.py --config_file config.json --uq_method sc --sc_q_order 3
```

### Command-Line Usage

Run a Monte Carlo simulation with 1000 samples:
```bash
python -m uqef.UQsim --uq_method mc --mc_numevaluations 1000 --parallel --num_cores 4
```

Run a Stochastic Collocation with the pseudo-spectral approach simulation with MPI:
```bash
mpirun -n 8 python your_script.py --uq_method sc --sc_q_order 3 --sc_p_order 2 --mpi
```

Run Saltelli sensitivity analysis:
```bash
python your_script.py --uq_method saltelli --mc_numevaluations 1000 --compute_Sobol_t
```

## UQsim parametrisation options

### UQ method and uncertain parameter settings
- `--uncertain`: Uncertain setting: can be evaluated to choose different probability distributions and their parameter values
- `--uq_method`: Define the UQ method: `sc`, `mc`, `saltelli`, or `ensemble`

#### Monte Carlo (`--uq_method mc`)
- `--mc_numevaluations`: Number of Monte Carlo samples
- `--sampling_rule`: Sampling strategy (`random`, `sobol`, `latin_hypercube`, `halton`, `hammersley`)
- `--regression`: Enable regression-based surrogate modeling (i.e., PCE-based)

#### Stochastic Collocation (`--uq_method sc`)
- `--sc_q_order`: Quadrature order (collocation points per dimension)
- `--sc_p_order`: Polynomial order (PCE terms)
- `--sc_quadrature_rule`: Quadrature rule (default: 'G' for Gaussian)
- `--sc_sparse_quadrature`: Enable sparse grid quadrature
- `--cross_truncation`: Cross-truncation parameter for polynomial basis

#### Saltelli (`--uq_method saltelli`)
- `--mc_numevaluations`: Number of base samples
- `--compute_Sobol_t`: Compute total Sobol indices
- `--compute_Sobol_m`: Compute main effect indices
- `--compute_Sobol_m2`: Compute second-order indices

#### Ensemble (`--uq_method ensemble`)
- `--read_nodes_from_file`: Read parameter values from file
- `--parameters_file`: File containing parameter sets

### Model and result directories
- `--inputModelDir`: Folder for the input files of the model
- `--outputModelDir`: Folder for the output files of the model
- `--outputResultDir`: Folder for the statistics results (plots, tables (csv), ...)

### Model settings
- `--model`: Name of the model
- `--model_variant`: Variant of the chosen model

### Parallelization Options
- `--parallel`: Enable shared-memory parallelization with threading
- `--num_cores`: Number of cores per node to use (default: all available)
- `--mpi`: Enable MPI parallelization
- `--mpi_method`: Choose MPI solver (`MpiPoolSolver` or `MpiSolver`)
- `--mpi_combined_parallel`: Enable hybrid MPI + multiprocessing (data distribution to the nodes via MPI and parallelisation with a node via threading)
- `--chunksize`: Number of runs that are chunked into a group
- `--mpi_chunksize`: Number of runs that are sent as a package via MPI

### Runtime Analysis and Optimization
- `--analyse_runtime`: Enable runtime analysis
- `--opt_runtime`: Enable runtime optimization with load balancing
- `--opt_runtime_gpce_Dir`: Define the folder for the runtime data
- `--opt_algorithm`: Scheduling algorithm (FCFS, LPT, SPT, or MULTIFIT)
- `--opt_strategy`: Optimization strategy (FIXED_ALTERNATE, FIXED_LINEAR, or DYNAMIC)

### Statistics Options
- `--disable_statistics`: Disable all statistical calculations including plots (useful when restoring a saved uqsim object from file)
- `--disable_calc_statistics`: Disable the calculation of statistics
- `--disable_recalc_statistics`: Disable the recalculation of statistics (useful when restoring a saved uqsim object from file)

### UQsim State Management: Save/Restore
- `--uqsim_store_to_file`: Save UQsim state for later restoration
- `--uqsim_restore_from_file`: Restore UQsim from saved state
- `--uqsim_file`: Filename for state storage (default: uqsim.saved)

### Additional Output Options
- `--save_all_simulations`: Save complete simulation data
- `--store_qoi_data_in_stat_dict`: Store quantity of interest data
- `--store_gpce_surrogate_in_stat_dict`: Store PCE surrogate model
- `--instantly_save_results_for_each_time_step`: Save results incrementally (has to be done in custom models)

## Project Structure

```
UQEF/
├── src/
│   └── uqef/
│       ├── __init__.py           # Package initialization
│       ├── UQsim.py              # Main simulation class
│       ├── model/                # Model definitions
│       ├── nodes/                # Parameter node management
│       ├── schedule/             # Task scheduling algorithms
│       ├── simulation/           # UQ method implementations
│       ├── solver/               # Parallel solvers
│       ├── stat/                 # Statistical analysis
│       └── util/                 # Utility functions
├── examples/                     # Example scripts
├── pyproject.toml                # Package setup
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── setup.cfg                     # Package setup configuration
```

## Examples

The `examples/` directory contains several demonstration scripts:

- **`simple_example_uqsim.py`**: Basic UQ simulation with the test model
- **`simple_example_testmodel.py`**: Direct model usage (without a UQsim object)
- **`simple_example_uqsim_config_file.py`**: Configuration file-based setup
- **`simple_example_uqsim_restore.py`**: State restoration example

To run an example:
```bash
cd examples
python simple_example_uqsim.py --uq_method sc --sc_q_order 3
```

Or using the provided shell script:
```bash
cd examples
bash start_simple_example_uqsim.sh
```

## Custom Model and Statistics Integration

To integrate your own model with UQEF:
- Create and register a custom model
- Create and register a custom statistics for your model

### Custom Model
1. Create a model class that inherits from `uqef.model.Model` (For a valid model implementation look at: [TestModel.py](src/uqef/model/TestModel.py))
2. Implement the required methods
3. Register your model in the `models` dictionary

Example for custom model usage:
```python
from CustomModel import CustomModel

# choose custom_model
uqsim.args.model = "custom_model"

# register model
uqsim.models.update({"custom_model": lambda: CustomModel()})
```

### Custom Statistics
1. Create a statistics class that inherits from `uqef.stat.Statistics`
2. Implement the required methods
3. Register your model in the `statistics` dictionary (For a valid model implementation look at: [TestModelStatistics.py](src/uqef/stat/TestModelStatistics.py))

Example for custom model and statistics usage:
```python
from CustomModel import CustomModel
from CustomStatistics import CustomStatistics

# choose custom_model
uqsim.args.model = "custom_model"

# register model
uqsim.models.update({"custom_model": lambda: CustomModel()})

# register statistics
uqsim.statistics.update({"custom_model": lambda: CustomStatistics()})
```

## Advanced Features

### Sparse Grid Quadrature
For high-dimensional problems, enable sparse grids* to reduce computational cost:
```bash
python your_script.py --uq_method sc --sc_sparse_quadrature --sc_q_order 5
```

*Here, the chaospy sparse grid implementation is used.

### Regression-Based PCE
Build surrogate models using regression instead of collocation:
```bash
python your_script.py --uq_method mc --regression
```

### Runtime Optimization for Heterogeneous Tasks
Enable intelligent load balancing for varying computational costs*:
```bash
python your_script.py --analyse_runtime --opt_runtime --opt_algorithm LPT --opt_strategy DYNAMIC
```

*On the first run, it saves the runtime predictor on `save_statistics()`, and on the second run it load from a file and use it for the prediction and optimisation step within UQEF.

## Performance Considerations

- For large-scale problems, use MPI parallelization on a cluster/HPC with `--mpi`
- Adjust `--chunksize` and `--mpi_chunksize` for optimal load balancing
- Enable `--analyse_runtime` and `--opt_runtime` for heterogeneous computational loads
- Use sparse quadrature for problems with dimension > 5
- Consider regression-based approaches for high-dimensional spaces

## Troubleshooting

### MPI Issues
If you encounter MPI-related errors:
```bash
# Check MPI installation
mpiexec --version

# Try running with explicit host specification
mpiexec -n 4 python your_script.py --mpi
```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please contact the author for guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for details.

## Author

**Florian Kuenzner**
Technical University of Munich (TUM), Rosenheim Technical University of Applied Sciences
Email: florian.kuenzner@th-rosenheim.de

**Ivana Jovanovic**
Technical University of Munich (TUM)
Email: ivana.jovanovic@tum.de

## Repository

GitLab: [https://github.com/flo2k/UQEF.git](https://github.com/flo2k/UQEF.git)

## Citation

If you use UQEF in your research, please cite:

```bibtex
@software{uqef,
  author = {Kuenzner, Florian},
  title = {UQEF: Uncertainty Quantification Execution Framework},
  version = {1.0},
  url = {https://github.com/flo2k/UQEF.git},
  institution = {Technical University of Munich, Rosenheim Technical University of Applied Sciences}
}
```

```bibtex
@phdthesis{dissertation,
  author = {Künzner, Florian},
  title = {Efficient non-intrusive uncertainty quantification for large-scale simulation scenarios},
  year = {2021},
  school = {Technische Universität München},
  url = {https://mediatum.ub.tum.de/1576066},
}
```

## Acknowledgments

UQEF builds upon several excellent open-source projects:
- **chaospy**: For polynomial chaos expansion functionality
- **mpi4py**: For MPI parallelization support
- **NumPy/SciPy**: For numerical computing foundations

## Version History

- **v1.0** (Current): Production-stable release with comprehensive UQ methods and parallel computing support

---

For more information, examples, and updates, visit the [GitLab repository](https://github.com/flo2k/UQEF.git).
