import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

def comparar_arboles_regresion(X, y, cv_folds):
    """
    Compara un árbol de decisión superficial vs uno profundo utilizando
    validación cruzada y devuelve los RMSE promedios y el modelo ganador.
    """
    # 1. Instanciar los modelos
    # Usamos random_state=42 para garantizar la reproducibilidad de los resultados,
    # ya que la construcción de los árboles tiene un componente aleatorio.
    tree_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_deep = DecisionTreeRegressor(max_depth=10, random_state=42)
    
    # 2. Evaluar con validación cruzada
    # Scikit-learn maximiza las métricas, por lo que el MSE se devuelve en negativo.
    scores_shallow = cross_val_score(tree_shallow, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    scores_deep = cross_val_score(tree_deep, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    
    # 3. Convertir de MSE negativo a RMSE positivo y calcular el promedio
    rmse_shallow = np.mean(np.sqrt(-scores_shallow))
    rmse_deep = np.mean(np.sqrt(-scores_deep))
    
    # 4. Determinar el mejor modelo (el que tenga el MENOR error)
    mejor = "Superficial" if rmse_shallow < rmse_deep else "Profundo"
    
    # 5. Retornar el diccionario con la estructura exacta solicitada
    return {
        'rmse_superficial': rmse_shallow,
        'rmse_profundo': rmse_deep,
        'mejor_modelo': mejor
    }
