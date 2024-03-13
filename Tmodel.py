from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, 'iris_model.joblib')

# Load the saved model
loaded_model = joblib.load('iris_model.joblib')

# Make predictions with the loaded model (using the test set as an example)
predictions = loaded_model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
