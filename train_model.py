"""Train a simple academic-risk classifier and save the model as advisor_model.pkl."""
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Demonstration dataset only. Replace with approved, anonymized real data for research use.
# Features: CGPA (0-5), attendance percentage (0-100), study hours/week, failed courses.
data = [
    [4.5,95,18,0],[4.1,90,15,0],[3.8,88,12,0],[3.6,82,10,0],
    [3.4,80,9,1],[3.2,75,8,1],[3.0,78,7,1],[2.8,72,6,1],
    [2.5,65,5,2],[2.3,60,4,2],[2.0,55,3,3],[1.8,50,3,4],
    [1.5,45,2,5],[1.2,40,1,6],[4.8,98,20,0],[3.9,70,11,1],
    [2.7,85,5,2],[3.5,92,8,0],[2.1,68,4,3],[1.9,35,2,5],
    [4.2,84,14,0],[3.1,58,7,2],[2.4,73,4,3],[1.6,62,2,4],
    [3.7,86,13,0],[2.9,80,6,1],[2.2,48,3,4],[4.0,93,16,0],
    [3.3,76,9,1],[1.4,30,1,6],[2.6,67,5,2],[3.0,88,6,1],
]
labels = [
    'Good standing','Good standing','Good standing','Good standing',
    'Needs improvement','Needs improvement','Needs improvement','Needs improvement',
    'At risk','At risk','At risk','At risk','At risk','At risk',
    'Good standing','Needs improvement','At risk','Good standing','At risk','At risk',
    'Good standing','At risk','At risk','At risk','Good standing','Needs improvement',
    'At risk','Good standing','Needs improvement','At risk','At risk','Needs improvement'
]
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25, random_state=42, stratify=labels
)
model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Demo holdout accuracy:', round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))
output = Path(__file__).with_name('advisor_model.pkl')
with output.open('wb') as f:
    pickle.dump(model, f)
print(f'Saved model to {output}')
