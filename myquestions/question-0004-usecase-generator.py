import numpy as np
import random
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Generador de Casos de Prueba
# ==========================================
def generar_caso_de_uso_evaluar_probabilidades():
    # 1. Generar la cantidad de pacientes (entre 500 y 1000)
    n_pacientes = random.randint(500, 1000)
    
    # 2. Generar y_true altamente desbalanceado (aprox 5% de positivos)
    # Usamos np.random.choice con probabilidades [0.95, 0.05] para clases 0 y 1
    y_true = np.random.choice([0, 1], size=n_pacientes, p=[0.95, 0.05])
    
    # Asegurarnos de que haya al menos un positivo y un negativo para que el AUC no falle
    if np.sum(y_true) == 0:
        y_true[random.randint(0, n_pacientes-1)] = 1
    if np.sum(y_true) == n_pacientes:
        y_true[random.randint(0, n_pacientes-1)] = 0
        
    # 3. Generar probabilidades (y_pred_proba) simulando un modelo real
    # Si es clase 1, probabilidad tiende a ser alta; si es 0, tiende a ser baja
    y_pred_proba = np.zeros(n_pacientes)
    for i in range(n_pacientes):
        if y_true[i] == 1:
            # Distribución sesgada hacia valores altos (ej. 0.6 a 0.99)
            y_pred_proba[i] = np.clip(np.random.normal(0.8, 0.15), 0.0, 1.0)
        else:
            # Distribución sesgada hacia valores bajos (ej. 0.01 a 0.4)
            y_pred_proba[i] = np.clip(np.random.normal(0.2, 0.2), 0.0, 1.0)
            
    # Redondear probabilidades a 3 decimales para mayor realismo
    y_pred_proba = np.round(y_pred_proba, 3)
    
    # 4. Construir el diccionario de input
    input_dict = {
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }
    
    # 5. Calcular el output esperado (Verdad matemática)
    auc_esperado = roc_auc_score(y_true, y_pred_proba)
    mascara_alta_confianza = y_pred_proba > 0.85
    conteo_esperado = int(np.sum(mascara_alta_confianza))
    
    expected_output = (auc_esperado, conteo_esperado)
    
    return input_dict, expected_output
  
# ==========================================
# 2. Script de Evaluación (Testing)
# ==========================================

if __name__ == "__main__":
    print("Iniciando generación y evaluación del Modelo de Probabilidades...\n")
    
    # Generamos los datos aleatorios
    inputs, expected_output = generar_caso_de_uso_evaluar_probabilidades()
    
    # Mostramos resumen de los datos generados
    pacientes_totales = len(inputs['y_true'])
    pacientes_enfermos = np.sum(inputs['y_true'])
    print(f"--- Resumen del Caso Generado ---")
    print(f"Total de pacientes: {pacientes_totales}")
    print(f"Pacientes realmente enfermos (Clase 1): {pacientes_enfermos} ({pacientes_enfermos/pacientes_totales:.1%})")
    print(f"---------------------------------\n")
    
    try:
        # Ejecutamos la función del estudiante
        student_output = evaluar_probabilidades(**inputs)
        
        # 1. Validar que retorna una tupla
        assert isinstance(student_output, tuple), f"Error: Se esperaba una tupla, se recibió {type(student_output)}."
        
        # 2. Validar que la tupla tiene exactamente 2 elementos
        assert len(student_output) == 2, f"Error: La tupla debe tener exactamente 2 elementos, tiene {len(student_output)}."
        
        auc_estudiante, conteo_estudiante = student_output
        auc_esperado, conteo_esperado = expected_output
        
        # 3. Validar tipos de datos dentro de la tupla
        assert isinstance(auc_estudiante, float), "Error: El primer elemento de la tupla (AUC) debe ser un float."
        assert isinstance(conteo_estudiante, int), "Error: El segundo elemento de la tupla (Conteo) debe ser un int."
        
        # 4. Validar precisión de los cálculos
        assert np.isclose(auc_estudiante, auc_esperado), \
            f"Error en AUC. Esperado: {auc_esperado}, Recibido: {auc_estudiante}"
            
        assert conteo_estudiante == conteo_esperado, \
            f"Error en conteo. Esperado: {conteo_esperado}, Recibido: {conteo_estudiante}"
            
        print("✅ ¡Prueba superada exitosamente!")
        print(f"-> ROC AUC Score: {auc_estudiante:.4f}")
        print(f"-> Predicciones con alta confianza (>0.85): {conteo_estudiante}")
        
    except AssertionError as e:
        print("❌ Error en la validación:")
        print(e)
    except Exception as e:
        print("❌ Error inesperado durante la ejecución del código del estudiante:", e)
