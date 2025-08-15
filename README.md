# Rapid Time Series Datasets Library

**Efficient Rust-powered preprocessing for time series datasets with Python bindings**

---

## Overview

Time series datasets are a common data format in many fields, such as finance and healthcare. They play an important role in many real-world machine learning applications.

They consist of sequences of data points collected over time, and are often used for tasks such as forecasting and classification.

However, these datasets usually require various preprocessing requirements, such as handling missing values, downsampling, normalization, and standardization before they can be used effectively for training models. These preprocessing steps can quickly become computationally expensive for larger datasets, especially when working in Python, where the performance is often not as good as in lower-level languages like Rust or C++.

This project delivers a high-performance Rust library for common time series preprocessing tasks, accessible seamlessly from Python via PyO3 bindings. Our goal is to provide a fast, memory-efficient pipeline for preparing time series data suitable for modern ML workflows.

Key features:

- Efficient Rust implementation of core preprocessing operations
- Zero-copy data sharing via `numpy` arrays for maximum performance
- Pipeline abstraction for chaining preprocessing steps
- Easy integration with PyTorch Lightning via a dedicated DataModule for streamlined model training

---

## Installation & Build Instructions

### Dependencies

**Rust-side (Cargo.toml):**

| Crate        | Version / Feature                                      |
| ------------ | ------------------------------------------------------ |
| `log`        | 0.4.27                                                 |
| `numpy`      | 0.25.0                                                 |
| `pyo3`       | { version = "0.25.0", features = ["auto-initialize"] } |
| `pyo3-log`   | 0.12.4                                                 |
| `ndarray`    | 0.16.1                                                 |
| `rand`       | 0.8                                                    |
| `visibility` | 0.1.1                                                  |

**Python-side (virtual environment):**

| Library             | Version |
| ------------------- | ------- |
| `numpy`             | 1.26.4  |
| `psutil`            | 6.0.0   |
| `pandas`            | 2.2.3   |
| `aeon`              | 1.1.0   |
| `torch`             | 2.2.2   |
| `pytorch_lightning` | 2.2.2   |

---

### Build Steps

1. Create a new Python virtual environment using the tool of your choice (e.g., `venv`, `conda`, etc.) and activate it.

2. Install the Python build tool **maturin** if you don't have it yet:

   ```bash
   pip install maturin
   ```

3. Navigate to the Rust source directory containing the `Cargo.toml` file:

   ```bash
   cd /source_code
   ```

4. Build and install the Rust extension into your Python environment:

   ```bash
   maturin develop -r
   ```

   This compiles the Rust code and makes it importable as a Python module.

5. Import and use the module in Python, for example:

   ```python
   from rust_time_series import (
       ForecastingDataSet,
       ClassificationDataSet,
       SplittingStrategy,
       ImputeStrategy,
   )
   ```

---

## Usage Guide

### Data Format

The library expects a **3D NumPy array** with the following dimensions:

- **First dimension**: Instances (samples)
- **Second dimension**: Timesteps (time points)
- **Third dimension**: Features (variables)

**Note**:

- Forecasting datasets typically have only one instance
- Classification datasets have multiple instances

### Dataset Types

#### 1. ForecastingDataSet

Used for time series forecasting tasks where you have a continuous time series that needs to be split temporally.

```python
import numpy as np
from rust_time_series import ForecastingDataSet, ImputeStrategy

# Create sample data (1 instance, 1000 timesteps, 3 features)
data = np.random.randint(0, 100, (1, 1000, 3)).astype(float)

# Initialize the dataset with train/val/test ratios
forecast_ds = ForecastingDataSet(
    data,           # 3D numpy array
    0.7,           # Training ratio
    0.2,           # Validation ratio
    0.1            # Test ratio
)

# Apply preprocessing pipeline (all methods expect for split() and collect() are optional)
forecast_ds.impute(ImputeStrategy.Median)    # Handle missing values
forecast_ds.downsample(2)                    # Reduce data points (factor of 2)
forecast_ds.split()                          # Temporal split (required before collect)
forecast_ds.normalize()                      # Min-max normalization [0,1]
forecast_ds.standardize()                    # Z-score standardization (mean=0, std=1)

# Collect results with sliding window parameters
result = forecast_ds.collect(
    3,  # Window size (lookback)
    1,  # Prediction horizon (lookahead)
    1   # Step size (stride)
)

# Result contains train/val/test splits as sliding windows
# Returns tuple: (train, val, test) where each is (X, y)
train_X, train_y = result[0]  # Training set
val_X, val_y = result[1]      # Validation set
test_X, test_y = result[2]    # Test set
```

#### 2. ClassificationDataSet

Used for time series classification tasks where you have multiple time series instances with corresponding labels.

```python
import numpy as np
from rust_time_series import ClassificationDataSet, SplittingStrategy, ImputeStrategy

# Create sample data (100 instances, 50 timesteps, 2 features)
dummy_data = np.random.randint(0, 100, (3, 100, 100)).astype(float)
labels = np.random.randint(0, 1, (100,)).astype(
    float
)  # Binary labels for classification

# Initialize the dataset
classification_ds = ClassificationDataSet(
    data,           # 3D numpy array
    labels,         # 1D numpy array of labels
    0.7,           # Training ratio
    0.2,           # Validation ratio
    0.1            # Test ratio
)

# Apply preprocessing pipeline (all methods expect for split() and collect() are optional)
classification_ds.impute(ImputeStrategy.ForwardFill)     # Handle missing values
classification_ds.downsample(2)                          # Reduce data points
classification_ds.split(SplittingStrategy.Random)        # Split strategy (required before collect)
classification_ds.normalize()                            # Min-max normalization
classification_ds.standardize()                          # Z-score standardization

# Collect results
result = classification_ds.collect()

# Result contains train/val/test splits in original format
# Returns tuple: (train, val, test) where each is (X, y)
train_X, train_y = result[0]  # Training set
val_X, val_y = result[1]      # Validation set
test_X, test_y = result[2]    # Test set
```

### Method Details

#### Imputation Strategies

Handle missing values (NaN) in your time series data:

```python
# Available strategies:
ImputeStrategy.Median        # Replace with median of feature column
ImputeStrategy.Mean          # Replace with mean of feature column
ImputeStrategy.ForwardFill   # Replace with last valid observation
ImputeStrategy.BackwardFill  # Replace with next valid observation
```

#### Downsampling

Reduce the number of data points by keeping every n-th point:

```python
dataset.downsample(factor)   # factor: integer > 1
# Example: factor=2 keeps every 2nd point, factor=3 keeps every 3rd point
```

#### Splitting Strategies

**For ClassificationDataSet:**

```python
SplittingStrategy.Random     # Randomly shuffle instances before splitting
SplittingStrategy.InOrder    # Split without shuffling (preserves order)
```

**For ForecastingDataSet:**

- Uses temporal splitting automatically (splits along time dimension)

#### Normalization & Standardization

**Normalization** (Min-Max scaling to [0,1]):

```python
dataset.normalize()  # x' = (x - min) / (max - min)
```

**Standardization** (Z-score scaling):

```python
dataset.standardize()  # x' = (x - mean) / std
```

**Important**: Statistics (min/max, mean/std) are computed from the training set and applied to all splits to prevent data leakage.

For more usage examples, please refer to the [example notebook](source_code/python/usage.ipynb).

### Pipeline Order

The preprocessing pipeline should follow this order:

1. **Constructor**: Create dataset instance with data and split ratios
2. **Impute**: Fill missing values (optional)
3. **Downsample**: Reduce data points (optional)
4. **Split**: Apply splitting strategy (**required before collect**)
5. **Normalize/Standardize**: Scale features (optional)
6. **Collect**: Retrieve final processed datasets

### Performance Considerations

- The library minimizes data copying through efficient memory management
- For `ForecastingDataSet`: Only one copy operation during final windowing
- For `ClassificationDataSet`: Data copying occurs during splitting
- Use array views where possible to maintain performance

---

## Project Structure & Documentation

- **/presentation**: Contains all LaTeX source files, figures, and compiled PDFs for both interim and final presentations.
- **/report**: Includes LaTeX source files, figures, and the final project report PDF.
- **/source_code/python**: Holds all Python scripts related to testing the Rust-Python bindings, example usage notebooks, and benchmarking scripts.
- **/source_code/src**: Contains the Rust source code implementing the core library functionality.
- **/source_code/tests**: Includes all unit tests for the Rust components found in the `src` directory.

---

## Contributors

- Amir Ali Aali (amir.ali.aali@rwth-aachen.de)
- Marius Kaufmann (marius.kaufmann@rwth-aachen.de)
- Kilian Braun (kilian.braun@rwth-aachen.de)
