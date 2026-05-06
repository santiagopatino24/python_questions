import numpy as np
from sklearn.cluster import KMeans

def segmentar_y_calcular_distancias(X, n_clusters):
    """
    Agrupa datos usando KMeans y calcula la distancia euclidiana 
    de cada cliente (punto) al centroide de su cluster asignado.
    """
    # 1. Instanciar el modelo con los parámetros requeridos.
    # Añadimos n_init='auto' para coincidir con la configuración de tu generador 
    # y evitar las advertencias de versiones recientes de scikit-learn.
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    
    # 2. Entrenar el modelo y obtener las etiquetas para cada punto
    labels = model.fit_predict(X)
    
    # 3. Obtener las coordenadas de los centroides finales
    centroids = model.cluster_centers_
    
    # 4. Calcular distancias de forma vectorizada (¡Mucho más rápido que un bucle for!)
    # Al hacer centroids[labels], creamos un array del mismo tamaño que X 
    # donde cada fila contiene el centroide exacto que le corresponde a ese punto.
    # Luego, np.linalg.norm calcula la distancia euclidiana entre ambas matrices fila por fila (axis=1).
    distancias = np.linalg.norm(X - centroids[labels], axis=1)
    
    # 5. Retornar la tupla con etiquetas y distancias
    return labels, distancias
