# Este es el bucle principal que impulsa el algoritmo MRL-AMIS.
# Integra el agente RL, las cajas de inteligencia, la decodificación, la evaluación y la actualización de la población

import numpy as np
import pandas as pd
import time
from pygmo import hypervolume
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inicialización antes del bucle
hypervolume_history = []  # Lista para guardar el historial de hipervolumen

def is_dominated(obj1, obj2, maximize_objectives_list):
    """
    Determina si obj1 está dominado por obj2.

    Argumentos:
        obj1 (np.ndarray): primer vector objetivo.
        obj2 (np.ndarray): segundo vector objetivo.
        maximize_objectives_list (lista): lista de valores booleanos que indican qué objetivos se deben maximizar.

    Devuelve:
        bool: verdadero si obj1 está dominado por obj2; falso en caso contrario
    """
    # Asegurarse de que obj1 y obj2 tengan las mismas dimensiones
    if len(obj1) != len(obj2) or len(obj1) != len(maximize_objectives_list):
        raise ValueError("Los vectores objetivo y la lista de maximización deben tener la misma longitud")

    at_least_one_better = False

    for i in range(len(obj1)):
        if maximize_objectives_list[i]:  # Objetivo de maximización
            if obj2[i] > obj1[i]:  # obj2 es mejor en este objetivo
                at_least_one_better = True
            elif obj2[i] < obj1[i]:  # obj1 es mejor en este objetivo.
                return False  # obj1 no está dominado
        else:  # Objetivo de minimización
            if obj2[i] < obj1[i]:  # obj2 es mejor en este objetivo
                at_least_one_better = True
            elif obj2[i] > obj1[i]:  # obj1 es mejor en este objetivo
                return False  # obj1 no está dominado

    # Si llegamos aquí, obj2 es al menos tan bueno como obj1 en todos los objetivos
    # obj1 está dominado por obj2 si obj2 es estrictamente mejor en al menos un objetivo
    return at_least_one_better


def find_pareto_front(objectives_np, maximize_objectives_list):
    """
    Identifica el frente de Pareto a partir de un conjunto de vectores objetivo utilizando comparaciones por pares.

    Argumentos:
        objectives_np (np.ndarray): Una matriz numpy de vectores objetivo (forma: num_solutions, num_objectives).
                                    Cada fila son los valores objetivo de una solución.
        maximize_objectives_list (lista): Una lista de valores booleanos que indican la dirección de optimización
                                         de cada objetivo (True para maximizar, False para minimizar).

    Devuelve:
        tupla: Una tupla que contiene:
            - pareto_front_objectives (np.ndarray): Vectores objetivo de las soluciones del frente de Pareto.
            - pareto_front_indices (np.ndarray): Índices de las soluciones del frente de Pareto en la matriz original objectives_np.
    """
    if objectives_np.shape[0] == 0:
        return np.array([]), np.array([])  # Devuelve vacío si no hay soluciones

    num_solutions = objectives_np.shape[0]
    # Supongamos que todas las soluciones son inicialmente no dominadas
    is_dominated_flag = np.zeros(num_solutions, dtype=bool)

    # Comprueba cada solución con respecto a todas las demás soluciones para determinar cuál es dominante
    for i in range(num_solutions):
        # Si la solución i ya está marcada como dominada, no es necesario comprobar si domina a otras
        if is_dominated_flag[i]:
            continue

        for j in range(num_solutions):
            if i == j or is_dominated_flag[j]:  # Evita las comparaciones entre soluciones similares y las soluciones ya dominantes
                continue

            # Comprueba si la solución j domina a la solución i
            if is_dominated(objectives_np[i], objectives_np[j], maximize_objectives_list):
                is_dominated_flag[i] = True
                break  # La solución i está dominada, pasa a la siguiente solución i

            # Comprueba si la solución i domina a la solución j
            elif is_dominated(objectives_np[j], objectives_np[i], maximize_objectives_list):
                is_dominated_flag[j] = True  # Marca j como dominado si i lo domina
                # No hay interrupción aquí, ya que continuamos comparando i con otras soluciones

    # El frente de Pareto está formado por soluciones que no están dominadas por ninguna otra solución
    pareto_front_indices = np.where(~is_dominated_flag)[0]
    pareto_front_objectives = objectives_np[pareto_front_indices]

    return pareto_front_objectives, pareto_front_indices


def calculate_hypervolume(objectives_np, reference_point, maximize_objectives_list):
    """
    Calcula el hipervolumen normalizado (rango 0-1) de un conjunto de objetivos.

    Argumentos:
        objectives_np (np.ndarray): Una matriz numpy de vectores objetivo (forma: num_solutions, num_objectives)
        reference_point (np.ndarray): punto de referencia (peores valores para cada objetivo)
        maximize_objectives_list (lista): lista de valores booleanos que indican qué objetivos maximizar

    Devuelve:
        float: valor de hipervolumen normalizado entre 0,0 y 1,0
    """
    if len(objectives_np) == 0:
        return 0.0  # El conjunto vacío tiene hipervolumen cero

    num_objectives = objectives_np.shape[1]

    # 1. Calcular los límites objetivos para la normalización
    min_bounds = np.min(objectives_np, axis=0)
    max_bounds = np.max(objectives_np, axis=0)

    # Evita la división por cero
    obj_range = max_bounds - min_bounds
    obj_range[obj_range == 0] = 1.0  # Evita la división por cero

    # 2. Normalizar los objetivos al rango [0,1]
    objectives_normalized = (objectives_np - min_bounds) / obj_range

    # 3. Convertir a problema de minimización (pygmo utiliza minimización)
    objectives_minimization = np.copy(objectives_normalized)
    for i, maximize in enumerate(maximize_objectives_list):
        if maximize:
            # Para objetivos de maximización, invertir: 1 - valor normalizado
            objectives_minimization[:, i] = 1.0 - objectives_normalized[:, i]

    # 4. Normalizar punto de referencia
    reference_normalized = (reference_point - min_bounds) / obj_range
    # Convertir a minimización
    reference_minimization = np.copy(reference_normalized)
    for i, maximize in enumerate(maximize_objectives_list):
        if maximize:
            reference_minimization[i] = 1.0 - reference_normalized[i]

    # 5. Asegurar que el punto de referencia sea peor que todas las soluciones
    for i in range(num_objectives):
        max_obj = np.max(objectives_minimization[:, i])
        reference_minimization[i] = max(reference_minimization[i], max_obj * 1.1)

    # 6. Calcular el hipervolumen utilizando pygmo
    try:
        hv_calculator = hypervolume(objectives_minimization)
        hv_value = hv_calculator.compute(reference_minimization)

        # 7. Normalizar al rango [0,1]
        # Calcular el hipervolumen máximo posible en este espacio normalizado
        max_possible_hv = np.prod(reference_minimization - np.zeros(num_objectives))

        # Devuelve el hipervolumen normalizado
        normalized_hv = hv_value / max_possible_hv if max_possible_hv > 0 else 0.0

        return min(1.0, max(0.0, normalized_hv))  # Asegurarse de que el valor esté entre [0,1]

    except Exception as e:
        print(f"Error calculating hypervolume: {e}")
        return 0.0


def spacing_metric(solutions, maximize_objectives_list):
    """
    Calcula la métrica de espaciado para medir la uniformidad de la distribución del frente de Pareto.

    Argumentos:
        solutions (np.ndarray): conjunto de vectores objetivo (soluciones no dominadas)
        maximize_objectives_list (list): lista de valores booleanos que indican qué objetivos se deben maximizar.

    Devuelve:
        float: valor de espaciado (cuanto menor sea, mejor será la uniformidad de la distribución).
    """
    if len(solutions) < 2:
        return 0.0

    # Normalizar los objetivos al rango [0,1] para un cálculo justo de la distancia
    min_vals = np.min(solutions, axis=0)
    max_vals = np.max(solutions, axis=0)

    # Evitar la división por cero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Normalizar
    solutions_norm = (solutions - min_vals) / ranges

    # Invert maximization objectives for uniform distance calculation
    for i, maximize in enumerate(maximize_objectives_list):
        if maximize:
            solutions_norm[:, i] = 1 - solutions_norm[:, i]

    # Calculate distances between each pair of solutions
    n = len(solutions_norm)
    distances = np.zeros(n)

    for i in range(n):
        # Find distance to closest neighbor
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((solutions_norm[i] - solutions_norm[j]) ** 2))
                min_dist = min(min_dist, dist)
        distances[i] = min_dist

    # Calculate standard deviation of distances
    mean_dist = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - mean_dist) ** 2) / n)

    return spacing

def visualize_pareto_clustering(pareto_front, pareto_objectives, maximize_objectives_list):
    """
    Visualiza el frente de Pareto con agrupación para entender la diversidad.

    Args:
        pareto_front (list): Lista de soluciones en el frente de Pareto
        pareto_objectives (list/ndarray): Lista de vectores de objetivos en el frente de Pareto
        maximize_objectives_list (list): Lista de booleanos indicando qué objetivos maximizar
    """
    # Verificar si hay suficientes soluciones para clustering
    if pareto_objectives is None or (isinstance(pareto_objectives, np.ndarray) and pareto_objectives.size == 0) or len(pareto_objectives) < 5:
        print("No hay suficientes soluciones en el frente de Pareto para análisis de clustering.")
        return

    try:

        # Convertir a numpy array para procesamiento
        obj_array = np.array(pareto_objectives)

        # Normalizar valores para comparación justa
        obj_min = np.min(obj_array, axis=0)
        obj_max = np.max(obj_array, axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0  # Evitar división por cero

        obj_normalized = (obj_array - obj_min) / obj_range

        # Para objetivos a minimizar, invertir valores normalizados
        for i, maximize in enumerate(maximize_objectives_list):
            if not maximize:
                obj_normalized[:, i] = 1.0 - obj_normalized[:, i]

        # Determinar número óptimo de clusters
        n_clusters = min(5, max(2, int(np.sqrt(len(obj_normalized) / 2))))

        # Aplicar K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(obj_normalized)

        # Crear visualización
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Análisis de Diversidad del Frente de Pareto', fontsize=16)

        # Gráfico 3D (si hay al menos 3 objetivos)
        if obj_array.shape[1] >= 3:
            ax = fig.add_subplot(221, projection='3d')
            scatter = ax.scatter(obj_array[:, 0], obj_array[:, 1], obj_array[:, 2],
                               c=clusters, cmap='viridis', s=80, alpha=0.8)

            # Etiquetas
            obj_names = ["Preferencia", "Costo", "CO2", "Sostenibilidad", "Riesgo"]
            dir_labels = ["MAX", "MIN"]

            ax.set_title('Clusters en el Espacio de Objetivos (3D)')
            ax.set_xlabel(f'{obj_names[0]} ({dir_labels[0 if maximize_objectives_list[0] else 1]})')
            ax.set_ylabel(f'{obj_names[1]} ({dir_labels[0 if maximize_objectives_list[1] else 1]})')
            ax.set_zlabel(f'{obj_names[2]} ({dir_labels[0 if maximize_objectives_list[2] else 1]})')

            # Añadir centroides
            centroids = kmeans.cluster_centers_
            # Desnormalizar centroides
            centroids_actual = np.zeros_like(centroids)
            for i, maximize in enumerate(maximize_objectives_list[:3]):  # Solo primeros 3 objetivos para 3D
                if not maximize:
                    centroids_actual[:, i] = obj_min[i] + (1 - centroids[:, i]) * obj_range[i]
                else:
                    centroids_actual[:, i] = obj_min[i] + centroids[:, i] * obj_range[i]

            ax.scatter(centroids_actual[:, 0], centroids_actual[:, 1], centroids_actual[:, 2],
                     c='red', marker='x', s=200, alpha=1.0, label='Centroides')
            ax.legend()

        # Gráficos 2D para pares importantes de objetivos
        # Preferencia vs Costo
        ax1 = fig.add_subplot(222)
        ax1.scatter(obj_array[:, 0], obj_array[:, 1], c=clusters, cmap='viridis', s=80, alpha=0.8)
        ax1.set_title('Preferencia vs Costo')
        ax1.set_xlabel(f'Preferencia ({dir_labels[0 if maximize_objectives_list[0] else 1]})')
        ax1.set_ylabel(f'Costo ({dir_labels[0 if maximize_objectives_list[1] else 1]})')
        ax1.grid(True)

        # Preferencia vs Sostenibilidad (si existe)
        if obj_array.shape[1] >= 4:
            ax2 = fig.add_subplot(223)
            ax2.scatter(obj_array[:, 0], obj_array[:, 3], c=clusters, cmap='viridis', s=80, alpha=0.8)
            ax2.set_title('Preferencia vs Sostenibilidad')
            ax2.set_xlabel(f'Preferencia ({dir_labels[0 if maximize_objectives_list[0] else 1]})')
            ax2.set_ylabel(f'Sostenibilidad ({dir_labels[0 if maximize_objectives_list[3] else 1]})')
            ax2.grid(True)

        # Costo vs Riesgo (si existe)
        if obj_array.shape[1] >= 5:
            ax3 = fig.add_subplot(224)
            ax3.scatter(obj_array[:, 1], obj_array[:, 4], c=clusters, cmap='viridis', s=80, alpha=0.8)
            ax3.set_title('Costo vs Riesgo')
            ax3.set_xlabel(f'Costo ({dir_labels[0 if maximize_objectives_list[1] else 1]})')
            ax3.set_ylabel(f'Riesgo ({dir_labels[0 if maximize_objectives_list[4] else 1]})')
            ax3.grid(True)

        # Añadir leyenda global
        plt.colorbar(scatter, ax=[ax for ax in fig.axes if ax.name != '3d'],
                   label='Cluster')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('pareto_clusters.png')
        print(f"Gráfico de clusters guardado como 'pareto_clusters.png'")

        # Análisis estadístico de clusters
        print("\n===== ANÁLISIS DE DIVERSIDAD DEL FRENTE DE PARETO =====")
        print(f"Número de clusters: {n_clusters}")
        print(f"Tamaño total del frente: {len(pareto_front)} soluciones")

        # Estadísticas por cluster
        obj_names_full = ["Preferencia Total", "Costo Total", "CO2 Total",
                         "Sostenibilidad Combinada", "Riesgo Total"]

        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_size = len(cluster_indices)
            print(f"\nCluster {i+1}: {cluster_size} soluciones ({cluster_size/len(clusters)*100:.1f}%)")

            # Estadísticas del cluster
            cluster_obj = obj_array[cluster_indices]
            cluster_mean = np.mean(cluster_obj, axis=0)
            cluster_std = np.std(cluster_obj, axis=0)

            # Imprimir estadísticas formateadas
            print("  Características del cluster:")
            for j, name in enumerate(obj_names_full):
                if j < len(cluster_mean):
                    print(f"  - {name}: {cluster_mean[j]:.2f} ± {cluster_std[j]:.2f} " +
                         f"[{dir_labels[0 if maximize_objectives_list[j] else 1]}]")

            # Solución representativa (más cercana al centroide)
            centroid = kmeans.cluster_centers_[i]
            distances = np.sqrt(np.sum((obj_normalized[cluster_indices] - centroid)**2, axis=1))
            representative_idx = cluster_indices[np.argmin(distances)]
            representative_solution = pareto_front[representative_idx]

            # Imprimir solución representativa
            print("  Solución representativa:")
            ruta = representative_solution.get('ruta_decodificada', [])
            if ruta:
                print(f"  - Ruta: {' -> '.join(map(str, ruta))}")

            objetivos = representative_solution.get('objetivos', {})
            if objetivos:
                print("  - Objetivos:")
                for key, value in objetivos.items():
                    print(f"    · {key}: {value}")

        # Análisis de variabilidad entre clusters
        print("\nVariabilidad entre clusters:")
        for j, name in enumerate(obj_names_full):
            if j < obj_array.shape[1]:
                cluster_means = [np.mean(obj_array[np.where(clusters == i)[0], j]) for i in range(n_clusters)]
                min_mean = min(cluster_means)
                max_mean = max(cluster_means)
                print(f"- {name}: Varía de {min_mean:.2f} a {max_mean:.2f} " +
                     f"(Δ = {max_mean-min_mean:.2f}, {(max_mean-min_mean)/min_mean*100:.1f}%)")

    except ImportError as e:
        print(f"Error: No se puede realizar clustering. Falta biblioteca necesaria: {e}")
        print("Instala scikit-learn con: pip install scikit-learn")
    except Exception as e:
        print(f"Error durante el análisis de clustering: {e}")


def inverted_generational_distance(solutions, reference_front, maximize_objectives_list):
    """
    Calcula la métrica de distancia generacional invertida (IGD).

    Argumentos:
        solutions (np.ndarray): Conjunto de vectores objetivo.
        reference_front (np.ndarray): Frente de Pareto de referencia (verdadero o aproximado).
        maximize_objectives_list (lista): Lista de valores booleanos que indican qué objetivos se deben maximizar.

    Devuelve:
        float: Valor IGD (cuanto menor, mejor)
    """
    if len(solutions) == 0 or len(reference_front) == 0:
        return float('inf')

    # Normalizar los objetivos al rango [0,1] para un cálculo justo de la distancia
    # Encontrar los valores mínimo y máximo en ambos conjuntos
    min_vals = np.min(np.vstack([solutions, reference_front]), axis=0)
    max_vals = np.max(np.vstack([solutions, reference_front]), axis=0)

    # Evita la división por cero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Normalizar
    solutions_norm = (solutions - min_vals) / ranges
    reference_norm = (reference_front - min_vals) / ranges

    # Invertir los objetivos de maximización para el cálculo de la distancia uniforme
    for i, maximize in enumerate(maximize_objectives_list):
        if maximize:
            solutions_norm[:, i] = 1 - solutions_norm[:, i]
            reference_norm[:, i] = 1 - reference_norm[:, i]

    # Calcular las distancias desde cada punto en el frente de referencia hasta el punto más cercano en las soluciones
    total_distance = 0.0
    for ref_point in reference_norm:
        # Calcular la distancia euclidiana a cada solución
        distances = np.sqrt(np.sum((solutions_norm - ref_point) ** 2, axis=1))
        # Encuentra la distancia mínima
        min_distance = np.min(distances)
        total_distance += min_distance

    # Distancia promedio
    return total_distance / len(reference_front)


def identify_non_dominated_fronts(objectives_np, maximize_objectives_list):
    """
    Identifica frentes de Pareto utilizando clasificación no dominada
    Argumentos:
        objectives_np (np.ndarray): Una matriz numpy de vectores objetivo (forma: num_solutions, num_objectives)
        maximize_objectives_list (lista): Lista de valores booleanos que indican qué objetivos maximizar
    Devuelve:
        list: Una lista de listas, donde cada lista interna contiene índices de soluciones en un frente
        list: Una lista de números de frente para cada solución
    """
    num_solutions = objectives_np.shape[0]
    if num_solutions == 0:
        return [], [] # Devuelve vacío si no hay soluciones

    # Lista para almacenar qué soluciones domina cada solución (índices en objectives_np)
    dominates_set = [set() for _ in range(num_solutions)]
    # Lista para almacenar cuántas soluciones dominan cada solución
    dominated_count = [0] * num_solutions
    # Lista para almacenar el número frontal de cada solución (inicializada en -1)
    front = [-1] * num_solutions
    # Lista de listas para almacenar soluciones en cada frente
    fronts = [[]] # Empezar por Frente 0

    # Encontar Frente 0
    for i in range(num_solutions):
        for j in range(num_solutions):
            if i == j:
                continue
            # Comprueba si la solución i domina a la solución j
            if is_dominated(objectives_np[j], objectives_np[i], maximize_objectives_list): # si j domina a i
                dominates_set[i].add(j)
            # Comprueba si la solución j domina a la solución i
            elif is_dominated(objectives_np[i], objectives_np[j], maximize_objectives_list): # si i domina a j
                dominated_count[i] += 1

        # Si ninguna solución domina i, pertenece al primer frente (Frente 0)
        if dominated_count[i] == 0:
            front[i] = 0
            fronts[0].append(i)

    # Identificar frentes posteriores
    current_front_index = 0
    # Continuar mientras el índice frontal actual sea válido Y el frente actual no esté vacío
    while current_front_index < len(fronts) and fronts[current_front_index]:
        next_front = []
        # Soluciones de proceso en el frente actual (índices en los objetivos originales_np)
        for i_original_index in fronts[current_front_index]:
            # Para cada solución en el frente actual, comprueba las soluciones que domina
            for j_original_index in dominates_set[i_original_index]:
                # Disminuir el recuento dominado para la solución j
                dominated_count[j_original_index] -= 1
                # Si el recuento dominado para j se convierte en 0, pertenece al siguiente frente
                if dominated_count[j_original_index] == 0:
                    front[j_original_index] = current_front_index + 1
                    next_front.append(j_original_index)

        # Incrementa el índice frontal para la siguiente iteración
        current_front_index += 1
        # Si el siguiente frente no está vacío, añadirlo a la lista de frentes
        if next_front:
            fronts.append(next_front)

    # Limpia cualquier frente vacío que quede atrás
    while fronts and not fronts[-1]:
        fronts.pop()

    return fronts, front # Devuelve los frentes (lista de listas de índices) y el número de frente para cada solución


def calculate_crowding_distance(objectives_np, front_indices):
    """
    Calcula la distancia de aglomeración para soluciones en un solo frente.

    Argumentos:
    objectives_np (np.ndarray): Una matriz numpy de vectores objetivo.
    front_indices (lista): Lista de índices de soluciones en el frente actual.

    Devuelve:
    lista: Una lista de distancias de aglomeración para soluciones en el frente, correspondiente a front_indices
    """
    if not front_indices:
        return []

    num_solutions_in_front = len(front_indices)
    num_objectives = objectives_np.shape[1]
    crowding_distances = [0.0] * num_solutions_in_front

    # Obtener los vectores objetivos para las soluciones en este frente
    front_objectives = objectives_np[front_indices]

    for obj_index in range(num_objectives):
        # Ordenar las soluciones iniciales según el objetivo actual
        sorted_indices_in_front = sorted(range(num_solutions_in_front), key=lambda i: front_objectives[i, obj_index])

        # Asignar distancia infinita a los puntos límite
        crowding_distances[sorted_indices_in_front[0]] = float('inf')
        crowding_distances[sorted_indices_in_front[num_solutions_in_front - 1]] = float('inf')

        # Calcular la distancia para los puntos intermedios
        # Comprobar si hay puntos intermedios y si el rango objetivo no es cero
        if num_solutions_in_front > 2:
            obj_min = front_objectives[sorted_indices_in_front[0], obj_index]
            obj_max = front_objectives[sorted_indices_in_front[num_solutions_in_front - 1], obj_index]
            obj_range = obj_max - obj_min

            if obj_range != 0:
                for i in range(1, num_solutions_in_front - 1):
                    crowding_distances[sorted_indices_in_front[i]] += (front_objectives[sorted_indices_in_front[i + 1], obj_index] - front_objectives[sorted_indices_in_front[i - 1], obj_index]) / obj_range
            # Si obj_range es 0, las distancias de aglomeración intermedias para este objetivo permanecen en 0,0

    return crowding_distances


def update_population(current_population_results, generated_solutions_this_iteration, population_size, maximize_objectives_list):
    """
    Selecciona la población para la siguiente iteración utilizando la selección multiobjetivo (clasificación no dominada y distancia de aglomeración).
    Argumentos:
        current_population_results (lista): Lista de diccionarios de la población actual.
        generated_solutions_this_iteration (lista): Lista de diccionarios para las soluciones recién generadas.
        population_size (int): El tamaño deseado de la población de la siguiente generación.
        maximize_objectives_list (list): Lista de booleanos que indican qué objetivos se deben maximizar.
    Devuelve:
        lista: La lista de diccionarios que representan la población de la siguiente generación.
    """
    # Combinar todas las soluciones de la población actual y las recién generadas
    all_solutions_combined = current_population_results + generated_solutions_this_iteration

    if not all_solutions_combined:
        return [] # Devuelve vacío si no hay soluciones

    # Extraer los objetivos y el estado de viabilidad de todas las soluciones combinadas
    all_objectives_dicts = [sol.get('objetivos', {}) for sol in all_solutions_combined]
    all_feasibility = [sol.get('is_feasible', False) for sol in all_solutions_combined]

    # Convertir los objetivos en una matriz numpy (solo para soluciones viables para el análisis de Pareto)
    feasible_indices_in_combined = [i for i, is_feat in enumerate(all_feasibility) if is_feat and all_solutions_combined[i].get('objetivos') is not None and all_solutions_combined[i].get('objetivos')]

    feasible_objectives_np = np.array([
        [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
         obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0), # Sostenibilidad combinada
         obj_dict.get('riesgo_total', 0)]
        for i in feasible_indices_in_combined
        for obj_dict in [all_objectives_dicts[i]] # Obtenga los objetivos dictados para esta solución viable
    ])

    # Realizar una clasificación no dominada de las soluciones viables
    feasible_fronts, feasible_ranks = identify_non_dominated_fronts(feasible_objectives_np, maximize_objectives_list)

    # Almacenar la clasificación y la distancia de aglomeración para todas las soluciones
    ranks = [-1] * len(all_solutions_combined) # Inicializar rangos a -1
    crowding_distances = [0.0] * len(all_solutions_combined) # Inicializar las distancias de aglomeración a 0.0

    # Asignar rangos y calcular la distancia de aglomeración para soluciones viables
    for front_rank, front_indices_in_feasible_np in enumerate(feasible_fronts):
        # Calcular la distancia de aglomeración para el frente viable actual
        front_objectives_np = feasible_objectives_np[front_indices_in_feasible_np]
        # Índices de paso relativos a front_objectives_np (0 a len(front_indices_in_feasible_np)-1)
        front_crowding_distances = calculate_crowding_distance(front_objectives_np, list(range(len(front_indices_in_feasible_np))))

        # Mapear de vuelta a los índices originales en all_solutions_combined
        for i, original_combined_index in enumerate([feasible_indices_in_combined[j] for j in front_indices_in_feasible_np]):
             ranks[original_combined_index] = front_rank
             crowding_distances[original_combined_index] = front_crowding_distances[i]

    # Asignar rango y distancia a soluciones inviables
    # Asignar un rango peor que cualquier solución viable (por ejemplo, número de frentes viables)
    worst_feasible_rank = len(feasible_fronts)
    for i in range(len(all_solutions_combined)):
        if not all_feasibility[i]:
            ranks[i] = worst_feasible_rank # Asignar una clasificación peor que cualquier frente viable
            crowding_distances[i] = 0.0 # Asignar diversidad mínima (más concurrido)

    # Crear una lista de tuplas (rango, -distancia_de_aglomeración, índice_original)
    # Utilizamos -distancia_de_aglomeración porque queremos maximizar la distancia de aglomeración durante la clasificación.
    # Cuanto menor sea el rango, mejor; cuanto mayor sea la distancia de aglomeración, mejor (de ahí el signo negativo).
    selection_criteria = [(ranks[i], -crowding_distances[i], i) for i in range(len(all_solutions_combined))]

    # Ordenar las soluciones según su rango (ascendente) y, a continuación, según la distancia de aglomeración descendente
    selection_criteria.sort()

    # Selecciona los individuos con mayor «tamaño de población».
    next_population_indices = [index for rank, neg_dist, index in selection_criteria[:population_size]]

    # Crear la siguiente lista de población utilizando los índices seleccionados
    next_population_results = [all_solutions_combined[i] for i in next_population_indices]

    return next_population_results


def calculate_set_coverage(set_a, set_b, maximize_objectives_list):
    """
    Calcula la métrica de cobertura del conjunto entre dos conjuntos de soluciones.

    Argumentos:
        set_a (np.ndarray): primer conjunto de vectores objetivo.
        set_b (np.ndarray): segundo conjunto de vectores objetivo.
        maximize_objectives_list (lista): lista de valores booleanos que indican qué objetivos se deben maximizar.

    Devuelve:
        float: proporción de soluciones en set_b dominadas por al menos una solución en set_a
    """
    if len(set_b) == 0:
        return 0.0

    dominated_count = 0
    for obj_b in set_b:
        for obj_a in set_a:
            if is_dominated(obj_b, obj_a, maximize_objectives_list):
                dominated_count += 1
                break  # Una vez dominado, no es necesario seguir comprobando

    return dominated_count / len(set_b)


def calculate_average_ratio_pareto(solutions_np, population_np, maximize_objectives_list):
    """
    Calcula la relación media de soluciones óptimas de Pareto.

    Argumentos:
        solutions_np (np.ndarray): Conjunto de soluciones no dominadas (vectores objetivo).
        population_np (np.ndarray): Población completa de soluciones (vectores objetivo).
        maximize_objectives_list (lista): Lista de valores booleanos que indican qué objetivos se deben maximizar.

    Devuelve:
        float: valor ARP (ratio de soluciones no dominadas en la población).
    """
    if len(population_np) == 0:
        return 0.0

    # Para el ARP estándar, solo necesitamos la proporción de soluciones no dominadas en la población
    return len(solutions_np) / len(population_np)