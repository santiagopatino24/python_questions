import pandas as pd
import numpy as np
import random
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. Generador de Casos de Prueba
# ==========================================
def generar_caso_de_uso_detectar_fallos_maquinaria():
    # 1. Determinar el tamaño de los datos
    n_rows = random.randint(200, 500)
    
    # 2. Generar datos "normales" (Ej: Máquina operando alrededor de 80°C con leve ruido)
    temperaturas = np.random.normal(loc=80.0, scale=2.5, size=n_rows)
    
    # 3. Inyectar anomalías aleatorias (picos de temperatura extremos)
    num_picos = int(n_rows * 0.05) # 5% de los datos serán picos extremos
    indices_picos = random.sample(range(n_rows), num_picos)
    for idx in indices_picos:
        temperaturas[idx] += random.choice([-25.0, 30.0]) # Caídas o subidas bruscas
        
    df = pd.DataFrame({'temperatura': np.round(temperaturas, 2)})
    
    # 4. Generar parámetros aleatorios
    tamano_ventana = random.randint(3, 8)
    tasa_contaminacion = round(random.uniform(0.05, 0.15), 2)
    
    # 5. Construir diccionario de input
    input_dict = {
        'df': df.copy(),
        'tamano_ventana': tamano_ventana,
        'tasa_contaminacion': tasa_contaminacion
    }
    
    # 6. Calcular el output esperado (Verdad matemática)
    if tamano_ventana > len(df):
        raise ValueError("La ventana es mayor que los datos disponibles")
        
    suavizado = df['temperatura'].rolling(window=tamano_ventana).mean().dropna()
    X_suavizado = suavizado.values.reshape(-1, 1)
    
    iso_forest = IsolationForest(contamination=tasa_contaminacion, random_state=42)
    predicciones = iso_forest.fit_predict(X_suavizado)
    
    conteo_anomalias = int(np.sum(predicciones == -1))
    tasa_real = float(conteo_anomalias / len(predicciones))
    
    expected_output = {
        "total_anomalias": conteo_anomalias,
        "tasa_real_anomalias": tasa_real
    }
    
    return input_dict, expected_output

# ==========================================
# 3. Script de Evaluación (Testing)
# ==========================================

if __name__ == "__main__":
    print("Iniciando generación y evaluación del Modelo de Anomalías...\n")
    
    # Generamos los datos aleatorios
    inputs, expected_output = generar_caso_de_uso_detectar_fallos_maquinaria()
    
    print(f"Parámetros del caso generado:\n- Filas: {len(inputs['df'])}\n- Ventana: {inputs['tamano_ventana']}\n- Contaminación: {inputs['tasa_contaminacion']}")
    
    try:
        # --- Prueba 1: Ejecución Normal ---
        student_output = detectar_fallos_maquinaria(**inputs)
        
        # Validamos estructura
        assert isinstance(student_output, dict), "Error: La función no devolvió un diccionario."
        assert "total_anomalias" in student_output and "tasa_real_anomalias" in student_output, "Error: Las claves del diccionario no son exactamente las solicitadas."
        
        # Validamos valores
        assert student_output["total_anomalias"] == expected_output["total_anomalias"], \
            f"Error en conteo. Esperado: {expected_output['total_anomalias']}, Recibido: {student_output['total_anomalias']}"
            
        assert np.isclose(student_output["tasa_real_anomalias"], expected_output["tasa_real_anomalias"]), \
            f"Error en tasa. Esperado: {expected_output['tasa_real_anomalias']}, Recibido: {student_output['tasa_real_anomalias']}"
        
        print("\n✅ Prueba 1 (Ejecución y Precisión Matemática) Superada.")
        
        # --- Prueba 2: Validación del ValueError ---
        print("\nEjecutando Prueba 2: Excepción de seguridad (Ventana gigante)...")
        try:
            detectar_fallos_maquinaria(inputs['df'], tamano_ventana=10000, tasa_contaminacion=0.1)
            print("❌ Prueba 2 Fallida: La función NO lanzó ningún error a pesar de que la ventana era más grande que el DataFrame.")
        except ValueError as ve:
            if "La ventana es mayor que los datos disponibles" in str(ve):
                print("✅ Prueba 2 Superada: El ValueError se activó con el mensaje exacto.")
            else:
                print(f"❌ Prueba 2 Fallida: Lanzó ValueError, pero el mensaje es distinto.\nRecibido: {ve}")
        except Exception as e:
            print(f"❌ Prueba 2 Fallida: Lanzó un error de tipo distinto al solicitado (Esperado: ValueError). Error: {type(e)}")

    except AssertionError as e:
        print("\n❌ Error en la validación:")
        print(e)
    except Exception as e:
        print("\n❌ Error inesperado durante la ejecución del código del estudiante:", e)
