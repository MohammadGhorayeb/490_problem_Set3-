
# Diabetes Model Project

This project implements a neural network model for predicting diabetes progression using the `DiabetesModel` class. The project includes a modular, reusable class designed to train, evaluate, and make predictions on any dataset with a similar structure to the [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) from `scikit-learn`.

## Project Structure

```
├── model_code
│   ├── diabetes_model_state.pth   # The saved model
│   ├── example_usage.py           # Demonstrates usage of the DiabetesModel class
│   └── model.py                   # Contains the DiabetesModel class
├── nn.ipynb                       # The model in a jupyter notebook format
├── README.md                      
└── requirements.txt


```

## Features

- **`DiabetesModel` Class**: Encapsulates the model, training, evaluation, and prediction methods.
- **Flexible Input**: The model dynamically adapts to the number of input features, making it compatible with various datasets.
- **Example Usage**: An example usage script demonstrates how to use the `DiabetesModel` class for training, evaluation, and prediction.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/MohammadGhorayeb/Diabetes.git
    cd Diabetes
    ```

2. **Install required dependencies** (e.g., `scikit-learn`, `torch`, `matplotlib`):
    ```bash
    pip install -r requirements.txt
    ```

   > **Note**: Ensure that `torch` is installed with CUDA support if you plan to use GPU acceleration. 

## Usage

1. **Define the Model Class**: The `DiabetesModel` class is defined in `model_code/diabetes_model.py`. You can import and initialize it by passing your dataset (`X` and `y`).

2. **Run the Example**:
   - An example script (`model_code/example_usage.py`) demonstrates a full pipeline:
     - Loading the dataset
     - Training the model
     - Evaluating the model on a test set
     - Making predictions

3. **Execute Example Usage**:
    ```bash
    python model_code/example_usage.py
    ```

## Code Walkthrough

### `diabetes_model.py`

The main components of the `DiabetesModel` class:

- **Initialization**: Handles data splitting, scaling, and conversion to PyTorch tensors.
- **`train` Method**: Trains the model with options for epochs and batch size, and saves a plot of training and validation losses.
- **`evaluate` Method**: Calculates and prints test loss and Mean Squared Error (MSE).
- **`predict` Method**: Takes input data and outputs predictions.

### `example_usage.py`

This script:
1. Loads the diabetes dataset.
2. Initializes the `DiabetesModel`.
3. Trains the model with the training and validation sets.
4. Evaluates the model with the test set.
5. Outputs predictions for new data samples.

## Example

Here is a basic example of how to use the `DiabetesModel` class:

```python
from sklearn.datasets import load_diabetes
from model_code.diabetes_model import DiabetesModel

# Load data
data = load_diabetes()
X = data.data
y = data.target

# Initialize model
model = DiabetesModel(X, y)

# Train model
model.train(num_epochs=100)

# Evaluate on test data
model.evaluate()

# Predict on test data
predictions = model.predict(model.X_test_tensor)
print(predictions[:5])
```

### Loading the Model's State Dictionary
To load the saved model, initialize the model architecture and load the parameters:

```python
# Initialize the model architecture with the same input size and parameters
model = DiabetesModel(X, y)

# Load the saved state dictionary
model.model.load_state_dict(torch.load('model_code/diabetes_model_state.pth'))
model.model.eval()  # Set to evaluation mode
```

## Requirements

- Python 3.6+
- PyTorch
- scikit-learn
- matplotlib

To install the required packages, you can use:

```bash
pip install torch scikit-learn matplotlib
```

