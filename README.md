# Function Approximation with PyTorch

This project implements various neural network models to demonstrate function approximation capabilities, exploring concepts of underfitting, optimal fitting, and overfitting.

## Project Structure

```
├── base/                    # Base classes for models and data loaders
├── configs/                 # Configuration files for different experiments
├── data_loader/            # Dataset implementation
├── model/                  # Model implementations
├── notebooks/              # Analysis notebooks
├── runs/                   # TensorBoard logs
└── saved/                  # Saved model checkpoints
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required dependencies:
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- TensorBoard >= 2.12.0
- Jupyter >= 1.0.0

## Implementation Details

### Dataset
- Implemented in `data_loader/function_dataset.py`
- Generates synthetic data for function approximation
- Supports different function types (linear, etc.)
- Configurable number of samples

### Model Architecture
- Base model implementation in `base/base_model.py`
- Dynamic neural network implementation in `model/dynamic_model.py`
- Configurable number of hidden layers and layer sizes
- Flexible activation functions

### Experiments

We conducted four main experiments to demonstrate different fitting scenarios:

1. **Basic Function Approximation**
   - 2 hidden layers [32, 16]
   - Baseline implementation

2. **Optimal Function Approximation**
   - 3 hidden layers [64, 32, 16]
   - Balanced architecture for optimal learning

3. **Underfitting Example**
   - 1 hidden layer [8]
   - Insufficient model capacity

4. **Overfitting Example**
   - 5 hidden layers [128, 128, 64, 64, 32]
   - Excessive model capacity

### Training Configuration

Each experiment uses specific hyperparameters defined in the configs:

- Optimizer: Adam
- Learning Rate: 0.01
- Loss Function: MSE
- Metrics: MAE, R² Score, Explained Variance
- Learning Rate Scheduler: StepLR (step_size=100, gamma=0.5)

## Results

The results for each experiment are saved in the `runs/` directory and can be visualized using TensorBoard. Model checkpoints are stored in the `saved/` directory.

Visualization examples:
- `runs/basic_approximation.png`
- `runs/optimal_approximation.png`
- `runs/overfit_approximation.png`
- `runs/underfit_approximation.png`

## Analysis

A detailed analysis of the experiments can be found in `notebooks/analysis.ipynb`, which includes:
- Training curves
- Model comparisons
- Performance metrics
- Visualization of predictions vs. ground truth

## Usage

To run the experiments:
```bash
python train.py -c configs/config.json     # Basic configuration
python train.py -c configs/optimal.json    # Optimal fitting
python train.py -c configs/underfit.json   # Underfitting example
python train.py -c configs/overfit.json    # Overfitting example
```

To run all experiments sequentially:
```bash
python run_experiments.py
```

## Conclusions

The experiments successfully demonstrate:
1. The impact of model architecture on function approximation
2. The trade-off between model complexity and performance
3. Practical examples of underfitting and overfitting
4. Optimal model selection strategies
