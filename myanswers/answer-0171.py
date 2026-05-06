import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def entrenar_modelo_riesgo(df, target_col):
    """
    Entrena un modelo de Regresión Logística escalando los datos previamente
    y calcula la exactitud (accuracy) sobre el mismo conjunto de datos.
    """
    # 1. Separar las características (X) y la variable objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Escalar X usando StandardScaler
    # Esto es crucial para la Regresión Logística porque asegura que 
    # variables grandes (como el colesterol) no dominen a las más pequeñas (como la edad).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Inicializar y entrenar el modelo
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # 4. Generar predicciones y calcular la exactitud (accuracy)
    predicciones = model.predict(X_scaled)
    exactitud = accuracy_score(y, predicciones)
    
    return exactitud
