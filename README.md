# moss
Solving engineering problems with evolutionary biology principles.

# Unit Tests
To run the unit tests for the `moss` package, you can use the following command in your terminal:

```bash
python -m pytest
```
This command will discover and execute all the test cases defined in the `tests` directory of the `moss` package. Make sure you have `pytest` installed in your Python environment. If you don't have it installed, you can do so using pip:

```bash
python -m pip install pytest
```

The environment used for development is `/run/media/jhello/8/python-envs/torch/bin/python`, e.g.:

```bash
/run/media/jhello/8/python-envs/torch/bin/python -m pytest -q -s
```

# DarwinSolver population settings

`DarwinSolver` (and `DarwinOptimizer`) accept the following keyword arguments. Every setting can be overridden in a unit test:

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `max_population` | `10000` | Maximum number of individuals per generation. After reproduction all candidates are scored and sorted; only the best `max_population` survivors may breed again, the rest are eliminated. |
| `offspring_per_individual` | `2` | Number of children each surviving individual breeds for the next generation. |
| `max_deviation` | `1.0` | Maximum per-coordinate mutation applied to each child. |
| `convergence_threshold` | `1e-6` | Score below which the optimization is considered converged. |
| `max_iterations` | `10000` | Maximum number of generations. |
| `seed` | `None` | RNG seed (set it for reproducible CPU/GPU comparisons). |
| `device` | auto | `'cpu'` (NumPy) or `'cuda'` (PyTorch). |

Example:

```python
solver = DarwinSolver(case, device='cpu', max_population=300,
                      offspring_per_individual=2, max_deviation=1.0,
                      convergence_threshold=1e-4, max_iterations=2000, seed=17)
solver.solve()
print(solver.iterations, solver.final_score)
```

# CPU vs GPU convergence

`tests/solver/Darwin/optimizer/test_optimizer.py::TestCpuGpuComparison` runs the same
seeded search on CPU and GPU and prints a comparison table. Measured on this machine
(NVIDIA GeForce 940MX, identical seed and parameters: `max_population=10000`,
`offspring_per_individual=2`, `max_deviation=1.0`, `convergence_threshold=1e-6`,
`max_iterations=2000`, `seed=42`):

| Device | Generations to converge | Final score | Wall time (s) |
| ------ | ----------------------- | ----------- | ------------- |
| CPU (NumPy) | 35 | 2.38e-07 | 0.647 |
| GPU (CUDA) | 27 | 4.77e-07 | 0.416 |

The GPU needs fewer generations in this run because it evaluates in `float32`,
which changes the trajectory slightly (the random mutations are identical on both
devices). Run the test yourself to reproduce:

```bash
/run/media/jhello/8/python-envs/torch/bin/python -m pytest -q -s \
  tests/solver/Darwin/optimizer/test_optimizer.py::TestCpuGpuComparison
```
