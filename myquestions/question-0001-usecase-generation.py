import pandas as pd
import numpy as np
import random
from numpy.testing import assert_array_almost_equal

# ==========================================
# 1. Generador de Casos de Prueba
# ==========================================
def generar_caso_de_uso_ingresos_por_categoria():
    # 1. Componente aleatorio: Definir categorías posibles
    categorias_posibles = ['Electrónica', 'Hogar', 'Deportes', 'Ropa', 'Alimentos', 'Ferretería', 'Juguetes']
    
    # 2. Generar el DataFrame 'df_catalogo' de forma aleatoria (entre 10 y 30 productos)
    num_productos = random.randint(10, 30)
    ids_catalogo = list(range(1000, 1000 + num_productos))
    
    df_catalogo = pd.DataFrame({
        'id_producto': ids_catalogo,
        # Precios aleatorios entre $5.0 y $500.0 con 2 decimales
        'precio_unitario': [round(random.uniform(5.0, 500.0), 2) for _ in range(num_productos)],
        'categoria_producto': [random.choice(categorias_posibles) for _ in range(num_productos)]
    })
    
    # 3. Generar el DataFrame 'df_ventas' de forma aleatoria (entre 50 y 150 ventas)
    num_ventas = random.randint(50, 150)
    df_ventas = pd.DataFrame({
        # Se escogen IDs que obligatoriamente existen en el catálogo para que el inner join funcione
        'id_producto': [random.choice(ids_catalogo) for _ in range(num_ventas)],
        'cantidad_vendida': [random.randint(1, 25) for _ in range(num_ventas)]
    })
    
    # 4. Construir el diccionario de input (las claves son los argumentos de la función)
    input_dict = {
        'df_ventas': df_ventas.copy(),
        'df_catalogo': df_catalogo.copy()
    }
    
    # 5. Calcular el output esperado (La verdad matemática o solución del problema)
    # Fusión (Inner Join)
    df_merged = pd.merge(df_ventas, df_catalogo, on='id_producto', how='inner')
    
    # Nueva columna
    df_merged['ingreso_total'] = df_merged['cantidad_vendida'] * df_merged['precio_unitario']
    
    # Agrupación y suma
    agrupado = df_merged.groupby('categoria_producto')['ingreso_total'].sum()
    
    # Ordenar de mayor a menor
    agrupado = agrupado.sort_values(ascending=False)
    
    # Extraer como arreglo de numpy
    expected_output = agrupado.values
    
    return input_dict, expected_output


# ==========================================
# 2. Script de Evaluación (Testing)
# ==========================================
if __name__ == "__main__":
    print("Iniciando generación de datos aleatorios...")
    
    # Llamada al generador (sin argumentos)
    inputs, expected_output = generar_caso_de_uso_ingresos_por_categoria()
    
    print("\n--- INPUT: Muestra de df_ventas ---")
    print(inputs['df_ventas'].head(3))
    print(f"(Total registros generados: {len(inputs['df_ventas'])})")
    
    print("\n--- INPUT: Muestra de df_catalogo ---")
    print(inputs['df_catalogo'].head(3))
    print(f"(Total productos en catálogo: {len(inputs['df_catalogo'])})")
    
    try:
        # Ejecutamos la función del estudiante desempaquetando el diccionario de inputs
        student_output = ingresos_por_categoria(**inputs)
        
        # Validaciones de tipo y exactitud matemática
        assert isinstance(student_output, np.ndarray), "Error: La función debe retornar un numpy.ndarray."
        assert_array_almost_equal(student_output, expected_output, decimal=2)
        
        print("\n✅ ¡Prueba superada exitosamente!")
        print("El código entregó los resultados esperados:")
        print(student_output)
        
    except AssertionError as e:
        print("\n❌ Error en la validación:")
        print(e)
        print("\nEsperado:", expected_output)
        print("Recibido:", student_output)
    except Exception as e:
        print("\n❌ Error durante la ejecución del código del estudiante:", e)
