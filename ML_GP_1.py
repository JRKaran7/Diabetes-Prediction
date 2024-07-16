import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Display class distribution
print(dataset['diabetes'].value_counts())

# Define features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Apply column transformer to encode categorical variables
ct = ColumnTransformer(transformers=[
    ('gender', OneHotEncoder(), [0]),
    ('smoking_history', OneHotEncoder(), [4])
], remainder='passthrough')

X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Apply scaling only after splitting the dataset to avoid data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Display class distribution after SMOTE
n = pd.DataFrame(y_train_sm).value_counts()
n.plot(kind='bar')
plt.show()

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(7.5, 4.5))
    sns.kdeplot(y_test, label='True')
    sns.kdeplot(y_pred, label='Predicted')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution of Original vs Predicted Classifications')
    plt.legend()
    plt.show()