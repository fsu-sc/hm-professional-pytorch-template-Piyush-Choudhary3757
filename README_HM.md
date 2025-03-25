# PyTorch Neural Network Template Implementation (60 Pts)

The objective of this homework is to implement a configurable neural network system using a professional PyTorch project template. You'll learn how to organize code in a maintainable structure while experimenting with neural network architectures.

## Setup

1. Clone your GitHub Classroom repository:

```bash
git clone YOUR_REPO_URL
```

2. Add the template repository as a remote and merge it into your repository:

```bash
git remote add template https://github.com/fsu-sc/ml_torch_templates.git
git fetch template
git merge template/main --allow-unrelated-histories
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

## 1. Implement Custom Dataset (10 pts)
Create `data_loader/function_dataset.py` that implements a custom dataset for function approximation. Your dataset should:

- Inherit from both `BaseDataLoader` and `torch.utils.data.Dataset`
- Support the following functions:
  - 'linear' → y = 1.5x + 0.3 + ε
  - 'quadratic' → y = 2x² + 0.5x + 0.3 + ε
  - 'harmonic' → y = 0.5x² + 5sin(x) + 3cos(3x) + 2 + ε
  - (where ε represents uniform distribution error between -1 and 1)
- Include proper data normalization (mean 0 and standard deviation 1)
- Be configurable through the config file

Values of x should be generated randomly between 0 and 2π.

Example structure:
```python
class FunctionDataset(torch.utils.data.Dataset):
    def init(self, n_samples=100, function='linear'):
    # Implementation here

class FunctionDataLoader(BaseDataLoader):
    def init(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
    self.dataset = FunctionDataset(n_samples, function)
    super().init(self.dataset, batch_size, shuffle, validation_split, num_workers)
```

## 2. Implement Model Architecture (10 pts)
Create `model/dynamic_model.py` that implements a configurable dense neural network. Your model should:

- Inherit from `BaseModel`
- Support:
  - Variable number of hidden layers (1-5)
  - Variable neurons per layer (1-100)
  - Multiple activation functions for hidden layers ('relu', 'sigmoid', 'tanh', 'linear')
  - Multiple activation functions for output layer ('relu', 'sigmoid', 'tanh', 'linear')
- Be configurable through the config file

## 3. Implement Training Metrics (10 pts)
Create `model/metric.py` to implement custom metrics for tracking model performance:

- Training loss
- Validation loss
- Custom metrics for function approximation accuracy

## 4. Configuration Files (10 pts)
Create multiple config files in the `configs/` directory for different experiments:

- Basic configuration (`configs/config.json`)
- Overfitting example (`configs/overfit.json`)
- Underfitting example (`configs/underfit.json`)
- Optimal configuration (`configs/optimal.json`)

## 5. TensorBoard Analysis (10 pts)
Use TensorBoard to compare your different model configurations:

1. Ensure all your training runs are properly logged using the template's TensorBoard integration
2. Compare and analyze:
   - Training time (epochs to convergence)
   - Training loss curves
   - Validation loss curves
   - Model architectures using `add_graph`
3. Include screenshots of your TensorBoard visualizations showing:
   - Side-by-side loss curve comparisons
   - Training speed differences
   - Architecture comparisons

## 6. Analysis and Documentation (10 pts)
Create a Jupyter notebook `notebooks/analysis.ipynb` that:

1. Loads different model configurations
2. Trains models using the template
3. Visualizes results
4. Analyzes overfitting and underfitting cases
5. Documents findings and conclusions

Include:
- Screenshots from your TensorBoard analysis
- Explanations of why certain configurations performed better/worse
- Recommendations for optimal configuration based on your findings

## Project Structure
Your final project should have the following structure:

```project/
│
├── data_loader/
│   └── function_dataset.py
├── model/
│   ├── dynamic_model.py
│   └── metric.py
├── configs/
│   ├── config.json
│   ├── overfit.json
│   ├── underfit.json
│   └── optimal.json
├── notebooks/
│   └── analysis.ipynb
└── runs/  # TensorBoard logs will be stored here
```

## Submission Guidelines
- Include your analysis notebook with all cells executed

