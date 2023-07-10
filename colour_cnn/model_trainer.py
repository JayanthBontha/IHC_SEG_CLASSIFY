import pandas as pd
df = pd.read_csv('finaldata_lab.csv')
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
X = df[['L', 'A', 'B']]
y = df['Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = joblib.load('./model/random_forest_model.pkl')


# Make predictions
y_pred = model.predict(X_test)
total = model.predict(X)
# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Confusion Matrix2:")
confusion = confusion_matrix(y, total)
print(confusion)

# joblib.dump(model, './model/random_forest_model.pkl')