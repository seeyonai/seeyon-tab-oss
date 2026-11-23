# Software and data provenance

Seeyon Tab's model, synthetic episode generator and training code are distributed under the [MIT license](LICENSE). The training presets initialize weights locally and draw every training example from the included generator.

The following libraries are installed separately. Each distribution carries its own license and third-party notices.

- NumPy supplies array operations and seeded random generation. [NumPy](https://github.com/numpy/numpy)
- MLX supplies the Apple Silicon training backend. [MLX](https://github.com/ml-explore/mlx)
- PyTorch supplies the CPU and CUDA training backend. [PyTorch](https://github.com/pytorch/pytorch)
