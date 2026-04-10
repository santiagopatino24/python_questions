import pandas as pd
import numpy as np
import random
from numpy.testing import assert_array_almost_equal

# ==========================================
# 1. Generador de Casos de Prueba
# ==========================================
def generar_caso_de_uso_agrupar_generaciones():
    # 1. Generar la cantidad aleatoria de usuarios
    num_usuarios = random.randint(100, 200)
    
    # 2. Crear el DataFrame con datos aleatorios
    df = pd.DataFrame({
        # Edades entre 1 y 99 años
        'edad': [random.randint(1, 99) for _ in range(num_usuarios)],
        # Días inactivo entre 0 (muy activo) y 60 (muy inactivo)
        'dias_inactivo': [random.randint(0, 60) for _ in range(num_usuarios)],
        # Tiempo en app entre 5 y 120 minutos
        'tiempo_en_app': [round(random.uniform(5.0, 120.0), 2) for _ in range(num_usuarios)]
    })
    
    # 3. Construir el diccionario de input
    input_dict = {
        'df': df.copy() # Pasamos una copia para no alterar el original
    }
    
    # 4. Calcular el output esperado (La lógica exacta del problema)
    # Filtrar usuarios con dias_inactivo <= 30
    df_filtrado = df[df['dias_inactivo'] <= 30].copy()
    
    # Crear la columna categórica
    df_filtrado['rango_edad'] = pd.cut(
        df_filtrado['edad'], 
        bins=[0, 18, 35, 60, 100], 
        labels=['Menor', 'Joven', 'Adulto', 'Mayor']
    )
    
    # Agrupar y promediar (observed=False evita un warning de Pandas con datos categóricos)
    promedios = df_filtrado.groupby('rango_edad', observed=False)['tiempo_en_app'].mean()
    
    # Extraer el array de numpy
    expected_output = promedios.values
    
    return input_dict, expected_output
  
# ==========================================
# 2. Script de Evaluación (Testing)
# ==========================================
if __name__ == "__main__":
    print("Iniciando generación y evaluación...")
    
    # Generamos los datos aleatorios
    inputs, expected_output = generar_caso_de_uso_agrupar_generaciones()
    
    print("\n--- INPUT: Muestra del DataFrame generado ---")
    print(inputs['df'].head())
    print(f"Total registros generados: {len(inputs['df'])}")
    
    try:
        # Ejecutamos la función del estudiante
        student_output = agrupar_generaciones(inputs['df'])
        
        # Validamos que devuelva un array de Numpy
        assert isinstance(student_output, np.ndarray), "Error: Se esperaba un arreglo de numpy (numpy.ndarray)."
        
        # Comparamos los arrays tolerando posibles diferencias ínfimas de decimales
        assert_array_almost_equal(student_output, expected_output, decimal=3)
        
        print("\n✅ ¡Prueba superada! El código del estudiante retornó:")
        print(student_output)
        
    except AssertionError as e:
        print("\n❌ Error en la validación:")
        print(e)
        print("\nEsperado:", expected_output)
        print("Recibido:", student_output)
    except Exception as e:
        print("\n❌ Error durante la ejecución del código del estudiante:", e)
