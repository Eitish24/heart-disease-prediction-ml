
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Column names for the dataset
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Load the dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=column_names, na_values='?')

# Drop rows with NaN values
df.dropna(inplace=True)
df = df.astype(float)

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='macro')
svm_recall = recall_score(y_test, svm_predictions, average='macro')
svm_f1 = f1_score(y_test, svm_predictions, average='macro')

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, average='macro')
dt_recall = recall_score(y_test, dt_predictions, average='macro')
dt_f1 = f1_score(y_test, dt_predictions, average='macro')

# Train KNN
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='macro')
knn_recall = recall_score(y_test, knn_predictions, average='macro')
knn_f1 = f1_score(y_test, knn_predictions, average='macro')

# Print performance
print("Support Vector Machine Performance:")
print(f"Accuracy: {svm_accuracy}, Precision: {svm_precision}, Recall: {svm_recall}, F1-Score: {svm_f1}")

print("\nDecision Tree Performance:")
print(f"Accuracy: {dt_accuracy}, Precision: {dt_precision}, Recall: {dt_recall}, F1-Score: {dt_f1}")

print("\nK-Nearest Neighbour Performance:")
print(f"Accuracy: {knn_accuracy}, Precision: {knn_precision}, Recall: {knn_recall}, F1-Score: {knn_f1}")

# Plotting
models = ['SVM', 'Decision Tree', 'KNN']
accuracy = [svm_accuracy, dt_accuracy, knn_accuracy]
precision = [svm_precision, dt_precision, knn_precision]
recall = [svm_recall, dt_recall, knn_recall]
f1_score_list = [svm_f1, dt_f1, knn_f1]
x = range(len(models))

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.bar(x, accuracy, color='blue')
plt.xticks(x, models)
plt.title('Accuracy')

plt.subplot(2, 2, 2)
plt.bar(x, precision, color='green')
plt.xticks(x, models)
plt.title('Precision')

plt.subplot(2, 2, 3)
plt.bar(x, recall, color='red')
plt.xticks(x, models)
plt.title('Recall')

plt.subplot(2, 2, 4)
plt.bar(x, f1_score_list, color='purple')
plt.xticks(x, models)
plt.title('F1-Score')

plt.tight_layout()
plt.show()
