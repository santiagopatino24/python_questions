import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def evaluar_overfitting(df, columnas, columna_objetivo, test_size):
    """
    Evalúa el sobreajuste de un modelo de regresión lineal calculando
    el MSE en los conjuntos de entrenamiento y prueba.
    """
    
    # 1. Extraer las variables predictoras (X) y la variable objetivo (y)
    X = df[columnas]
    y = df[columna_objetivo]
    
    # 2. Dividir los datos en entrenamiento y prueba. 
    # Es vital usar random_state=42 para que el resultado coincida exactamente 
    # con el generador de casos de uso durante la validación.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 3. Inicializar y entrenar el modelo de Regresión Lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Generar las predicciones para ambos conjuntos
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 5. Calcular el Error Cuadrático Medio (MSE)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    # 6. Devolver el resultado como un array de numpy
    return np.array([mse_train, mse_test])
