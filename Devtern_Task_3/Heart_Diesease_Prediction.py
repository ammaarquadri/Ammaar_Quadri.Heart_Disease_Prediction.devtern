## import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Switching matplotlib backend to TkAgg (if necessary)
import matplotlib
matplotlib.use('TkAgg')

# Load the dataset
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Data Cleaning and Preprocessing

# Check for missing values
missing_values = data.isnull().sum()
print(f"Missing values in each column:\n{missing_values}\n")

# Handle missing values by imputing with mean (you can also choose median or drop rows)
# Assuming here that all columns except 'Heart Disease' are numeric and need imputation
numeric_columns = data.select_dtypes(include='number').columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Check again for any remaining missing values
missing_values_after_impute = data.isnull().sum()
print(f"Missing values after imputation:\n{missing_values_after_impute}\n")

# Remove duplicates if any
data.drop_duplicates(inplace=True)

# Ensure 'target' column name matches the dataset
target_column = 'Heart Disease'

# Split the data into features (X) and target (y)
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}\n')
print(f'Confusion Matrix:\n{confusion}\n')
print(f'Classification Report:\n{report}\n')

# Analyzing Model Coefficients
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(f'Logistic Regression Model Coefficients:\n{coefficients}\n')

# Example: Visualizing the distribution of 'Heart Disease'
plt.figure(figsize=(6, 4))
sns.countplot(x='Heart Disease', data=data)
plt.title('Distribution of Heart Disease')
plt.xlabel('Presence or Absence')
plt.ylabel('Count')
plt.show()


# Plotting model coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y=coefficients.index, data=coefficients)
plt.title('Logistic Regression Model Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
