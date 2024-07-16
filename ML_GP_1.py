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

dataset = pd.read_csv('diabetes.csv')

print(dataset['Outcome'].value_counts())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset.head())
print(dataset.info())
print(dataset.describe())


outcome = dataset["Outcome"].value_counts()
labels = {0: 'Not Diabetes', 1: 'Diabetes'}
outcome.index = outcome.index.map(labels)
plt.pie(outcome, autopct="%1.1f%%")
plt.title("Diabetes Prediction")
plt.legend(title="Diabetes Prediction", labels=outcome.index)
plt.show()

db = ["BMI", "DiabetesPedigreeFunction", "Insulin", "Age"]

for i in range(len(db)):
    plt.hist(dataset[db[i]])
    plt.xlabel(db[i])
    plt.ylabel("values")
    plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='magma')
plt.title('Correlation Heatmap')
plt.show()

features = ["BMI", "DiabetesPedigreeFunction", "Insulin", "Age", "Pregnancies", "BloodPressure","Glucose","SkinThickness"]

for feature in features:
    sns.boxplot(x='Outcome', y=feature, data=dataset, palette='coolwarm')
    plt.title(f'Box Plot of {feature} by Outcome')
    plt.show()

sns.lineplot(x="Age", y="Glucose", hue='Outcome', data=dataset, palette='coolwarm')
plt.title(f'Line Graph of Age by Outcome')
plt.xlabel("Age")
plt.ylabel('Glucose')
plt.show()


sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=dataset, palette='coolwarm')
plt.title('Scatter Plot of BMI vs Glucose')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.show()

sns.violinplot(x='Outcome', y='BloodPressure', data=dataset, palette='coolwarm')
plt.title('Violin Plot of Blood Pressure by Outcome')
plt.show()


sns.kdeplot(data=dataset, x='Age', hue='Outcome', fill=True, palette='coolwarm')
plt.title('KDE Plot of Age Distribution by Outcome')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

for feature in features:
    sns.kdeplot(data=dataset, x=feature, hue='Outcome', fill=True, palette='coolwarm')
    plt.title(f'KDE Plot of {feature} by Outcome')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

n = pd.DataFrame(y_train_sm).value_counts()
n.plot(kind='bar')
plt.show()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

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
