
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
users_behavior = pd.read_csv('users_behavior.csv')

# Display the first 5 rows of the dataset to understand its structure
print("First five rows of users_behavior:")
print(users_behavior.head(5))

# Display information about the dataset, including data types and missing values
print("\nDataset information:")
print(users_behavior.info())

# Check for duplicated rows
print("Duplicated rows in users_behavior:")
print(users_behavior[users_behavior.duplicated()])

# Plot histograms to visualize the distribution of various features (calls, minutes, messages, mb_used)
plt.figure(figsize=(12, 8))

# Calls distribution
plt.subplot(2, 2, 1)
sns.histplot(data=users_behavior, x='calls', hue='is_ultra', palette='viridis', kde=True)
plt.title('Calls Distribution')

# Minutes distribution
plt.subplot(2, 2, 2)
sns.histplot(data=users_behavior, x='minutes', hue='is_ultra', palette='viridis', kde=True)
plt.title('Minutes Distribution')

# Messages distribution
plt.subplot(2, 2, 3)
sns.histplot(data=users_behavior, x='messages', hue='is_ultra', palette='viridis', kde=True)
plt.title('Messages Distribution')

# MB used distribution
plt.subplot(2, 2, 4)
sns.histplot(data=users_behavior, x='mb_used', hue='is_ultra', palette='viridis', kde=True)
plt.title('MB Used Distribution')

# Display the plots
plt.tight_layout()
plt.show()

# Separate the dataset into features (X) and target variable (y)
X = users_behavior[['calls', 'minutes', 'messages', 'mb_used']]
y = users_behavior['is_ultra']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier on the training data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Perform cross-validation using KFold (5 folds) and compute the average accuracy
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
average_cv_accuracy = cv_scores.mean()
print("Average cross-validation accuracy:", average_cv_accuracy)

# Additional task: Refit RandomForestClassifier and validate using cross-validation
X_new = users_behavior[['calls', 'minutes', 'messages', 'mb_used']]
y_new = users_behavior['is_ultra']
rf_model_new = RandomForestClassifier(random_state=42)

# Perform cross-validation again on the updated model
kf_new = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_new = cross_val_score(rf_model_new, X_new, y_new, cv=kf_new, scoring='accuracy')

# Calculate and print the average accuracy of the new cross-validation
average_accuracy_new = cv_scores_new.mean()
print("Average cross-validation accuracy with updated model:", average_accuracy_new)