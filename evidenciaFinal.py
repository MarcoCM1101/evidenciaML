# A01753729 Marco Antonio Caudillo Morales
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Función para cargar y limpiar los datos


def dataClean(path):
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(path)

    # Convertir la columna 'loan_status' a valores binarios (1 para "Approved", 0 para "Rejected")
    df[' loan_status'] = df[' loan_status'].replace(
        {' Approved': 1, ' Rejected': 0})

    # Seleccionar las características relevantes y la variable objetivo
    X = df[[' cibil_score', ' loan_term', ' no_of_dependents']]
    y = df[' loan_status']

    # Calcular la media y la desviación estándar de las características para estandarizarlas
    means = X.mean(axis=0).values  # Convertir a un array de numpy
    stds = X.std(axis=0).values  # Convertir a un array de numpy

    # Estandarizar las características (restar la media y dividir por la desviación estándar)
    X = (X - means) / stds

    # Añadir una columna de unos para el término de sesgo (bias) en el modelo de regresión logística
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Devolver los conjuntos de datos de entrenamiento y prueba, así como las medias y desviaciones estándar
    return X_train, X_test, y_train, y_test, means, stds

# Función sigmoide


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función para calcular el costo (error) del modelo en función de los parámetros actuales


def compute_cost(X, y, theta):
    m = len(y)  # Número de ejemplos de entrenamiento
    # Calcular las predicciones del modelo usando la función sigmoide
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5  # Pequeño valor para evitar errores de logaritmo de 0
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y)
                           * np.log(1 - h + epsilon))  # Función de costo
    return cost

# Función de gradiente descendente para optimizar los parámetros del modelo (theta)


def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)  # Número de ejemplos de entrenamiento
    cost_history = []  # Lista para almacenar la evolución del costo durante las iteraciones
    print('----------------------------------------------')

    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))  # Calcular las predicciones actuales
        # Calcular el gradiente de la función de costo
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient  # Actualizar los parámetros usando el gradiente
        # Calcular el costo después de la actualización
        cost = compute_cost(X, y, theta)
        # Almacenar el costo actual en la historia del costo
        cost_history.append(cost)

        # Imprimir el costo cada 1000 iteraciones para monitorear la convergencia
        if i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')
    print('----------------------------------------------')

    return theta, cost_history

# Función para realizar predicciones con el modelo entrenado


def predict(X, theta):
    # Calcular la probabilidad usando la función sigmoide
    probability = sigmoid(np.dot(X, theta))
    # Convertir la probabilidad en una clase (1 o 0)
    return [1 if x >= 0.5 else 0 for x in probability]


def main():
    print("\n------- Modelo de Regresión Logística --------")

    # Cargar y limpiar los datos desde el archivo CSV
    X_train, X_test, y_train, y_test, means, stds = dataClean(
        'loan_approval_dataset.csv')

    # Inicializar los pesos (theta) a cero
    weights = np.zeros(X_train.shape[1])

    # Entrenar el modelo usando gradiente descendente
    weights, cost_history = gradient_descent(
        X_train, y_train, weights, 0.01, 10000)

    # Realizar predicciones en el conjunto de prueba
    y_pred = predict(X_test, weights)

    # Evaluar el rendimiento del modelo usando una matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print('----------------------------------------------')
    print("Matriz de confusión:")
    print(cm)
    print('----------------------------------------------')

    # Calcular la precisión del modelo y mostrarla en consola
    accuracy = np.mean(y_pred == y_test) * 100
    print('----------------------------------------------')
    print(f"Precisión del modelo: {accuracy:.2f}%")
    print('----------------------------------------------')

    # Inputs de nuevos datos para realizar una predicción
    print("\n--- Predicción de Aprobación de Préstamo ---")
    cibil_score = float(input("Ingrese el puntaje CIBIL: "))
    loan_term = float(input("Ingrese el plazo del préstamo (en meses): "))
    no_of_dependents = float(input("Ingrese el número de dependientes: "))

    # Crear un nuevo vector de características y estandarizarlo usando las medias y desviaciones estándar del conjunto de entrenamiento
    new_data = np.array([[cibil_score, loan_term, no_of_dependents]])
    new_data = (new_data - means) / stds
    # Añadir el término de sesgo
    new_data = np.hstack((np.ones((new_data.shape[0], 1)), new_data))

    # Realizar la predicción con el nuevo dato
    new_prediction = predict(new_data, weights)

    # Mostrar el resultado de la predicción
    result = "Approved" if new_prediction[0] == 1 else "Rejected"
    print(f'\nPredicción para el nuevo dato: {result}')


if __name__ == '__main__':
    main()
