from utils import db_connect
engine = db_connect()

# your code here
# Paso 1: Carga del conjunto de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

print(df.head())
df.info()

# Paso 2: Realiza un EDA completo
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["Glucose"], kde=True)
plt.title("Distribución de Glucosa en Plasma")
plt.show()

print("Valores nulos por columna:")
print(df.isnull().sum())

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")

# Paso 3: Construye un modelo de árbol de decisión
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy:.2f}")

# Comparación de funciones de pureza
criteria = ["gini", "entropy"]
results = {}

for criterion in criteria:
    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[criterion] = accuracy_score(y_test, y_pred)

plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.xlabel("Criterio de pureza")
plt.ylabel("Exactitud")
plt.title("Comparación de Criterios de Pureza")
plt.show()

# Paso 4: Optimiza el modelo anterior
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Exactitud del modelo optimizado: {accuracy_best:.2f}")

# Paso 5: Guarda el modelo
joblib.dump(best_model, "decision_tree_model.pkl")
print("Modelo guardado como decision_tree_model.pkl")