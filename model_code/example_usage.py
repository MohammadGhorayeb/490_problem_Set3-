from sklearn.datasets import load_diabetes
from model import DiabetesModel

# Load the diabetes dataset (or any other dataset with similar structure)
data = load_diabetes()
X = data.data
y = data.target

# Initialize the model with the data
model = DiabetesModel(X, y)

# Train the model
model.train(num_epochs=100)

# Evaluate the model on the test data
model.evaluate()

# Make predictions on test data (example)
predictions = model.predict(model.X_test_tensor)
print(predictions[:5])  # Print first 5 predictions