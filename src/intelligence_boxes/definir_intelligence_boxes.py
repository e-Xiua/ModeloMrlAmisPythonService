# Intelligence Boxes estandarizados con una interfaz común

import random
import numpy as np

from estado_y_recompensa_rl.definir_comportamiento import is_dominated
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo, is_compatible_with_preferences
from generacionWorkPackages.workPackages import decodificar_wp

def ib_random_perturbation(wp_vector, **kwargs):
    """
    Caja de inteligencia: Perturba aleatoriamente los elementos del vector WP.
    Modifica el vector añadiendo un pequeño valor aleatorio a algunos elementos.
    Esto cambia los valores del vector, lo que afecta al orden de decodificación.

    Argumentos:
    wp_vector: Vector de trabajo a modificar
    **kwargs: Parámetros adicionales (ignorados en esta función)

    Devuelve:
    Vector de trabajo modificado
    """
    perturbation_strength = kwargs.get('perturbation_strength', 0.1)

    modified_wp_vector = wp_vector.copy()
    num_elements = len(modified_wp_vector)

    if num_elements > 0:
        # Determinar cuántos elementos perturbar (por ejemplo, un pequeño porcentaje).
        num_to_perturb = max(1, int(num_elements * 0.05))
        num_to_perturb = min(num_to_perturb, num_elements)

        indices_to_perturb = np.random.choice(num_elements, size=num_to_perturb, replace=False)

        # Añadir ruido aleatorio (positivo o negativo)
        perturbation = (np.random.rand(num_to_perturb) * 2 - 1) * perturbation_strength
        modified_wp_vector[indices_to_perturb] += perturbation

        # Asegurarse de que los valores se mantengan dentro del rango válido [0, 1] después de la perturbación.
        modified_wp_vector = np.clip(modified_wp_vector, 0, 1)

    return modified_wp_vector

def ib_swap_mutation(wp_vector, **kwargs):
    """
    Caja de inteligencia: Intercambia dos elementos en el vector WP.
    Cambia el orden relativo de dos POI intercambiando valores.

    Argumentos:
        wp_vector: Vector de trabajo a modificar
        **kwargs: Parámetros adicionales (ignorados en esta función)

    Devuelve:
        Vector de trabajo modificado
    """
    modified_wp_vector = wp_vector.copy()
    num_elements = len(modified_wp_vector)

    if num_elements >= 2:
        idx1, idx2 = np.random.choice(num_elements, size=2, replace=False)

        # Swap the values
        modified_wp_vector[idx1], modified_wp_vector[idx2] = modified_wp_vector[idx2], modified_wp_vector[idx1]

    return modified_wp_vector

def ib_inversion_mutation(wp_vector, **kwargs):
    """
    Caja de inteligencia: Invierte una sección del vector WP.
    Invierte el orden de una subsecuencia de POI en el orden de decodificación.

    Argumentos:
        wp_vector: Vector de trabajo a modificar.
        **kwargs: Parámetros adicionales (ignorados en esta función).

    Devuelve:
        Vector de trabajo modificado
    """
    modified_wp_vector = wp_vector.copy()
    num_elements = len(modified_wp_vector)

    if num_elements >= 2:
        start_idx = np.random.randint(0, num_elements)
        end_idx = np.random.randint(0, num_elements)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        if end_idx - start_idx >= 2:
             modified_wp_vector[start_idx:end_idx] = modified_wp_vector[start_idx:end_idx][::-1]

    return modified_wp_vector

def ib_guided_perturbation(wp_vector, **kwargs):
    """
    Perturba el vector de trabajo utilizando información del frente de Pareto actual
    """
    modified_wp_vector = wp_vector.copy()

    # Obtener datos necesarios
    current_population_results = kwargs.get('current_population_results', [])
    current_pareto_front_indices = kwargs.get('current_pareto_front_indices', [])

    # Si tenemos información del frente de Pareto y la población actual
    if current_population_results and current_pareto_front_indices and len(current_pareto_front_indices) > 0:
        # Seleccionar una solución aleatoria del frente de Pareto
        pareto_index = np.random.choice(current_pareto_front_indices)
        # Obtener el vector WP correspondiente
        if pareto_index < len(current_population_results):
            pareto_solution = current_population_results[pareto_index].get('wp_original', wp_vector)

            # Intensidad de la perturbación (0 a 1)
            strength = 0.3

            # Número de dimensiones a perturbar (como máximo el tamaño del vector más pequeño)
            max_dims = min(len(modified_wp_vector), len(pareto_solution))
            num_dimensions_to_perturb = min(3, max_dims)

            if num_dimensions_to_perturb > 0:
                # Seleccionar dimensiones aleatorias para perturbar
                dimensions_to_perturb = np.random.choice(
                    range(max_dims),
                    size=num_dimensions_to_perturb,
                    replace=False
                )

                for idx in dimensions_to_perturb:
                    # Mover hacia la solución Pareto con una magnitud aleatoria
                    direction = np.random.rand() * strength
                    modified_wp_vector[idx] = (1 - direction) * modified_wp_vector[idx] + direction * pareto_solution[idx]
    else:
        # Perturbación aleatoria normal si no hay información del Pareto
        # Seleccionar dimensiones aleatorias para perturbar
        num_dimensions_to_perturb = min(3, len(modified_wp_vector))

        if num_dimensions_to_perturb > 0:
            dimensions_to_perturb = np.random.choice(
                range(len(modified_wp_vector)),
                size=num_dimensions_to_perturb,
                replace=False
            )

            for idx in dimensions_to_perturb:
                # Pequeña perturbación aleatoria
                modified_wp_vector[idx] = modified_wp_vector[idx] * (1 + np.random.normal(0, 0.1))

    return modified_wp_vector

def ib_local_search(wp_vector, **kwargs):
    """
    Realiza una búsqueda local evaluando pequeños cambios en el vector WP.
    Elige el mejor cambio encontrado según evaluación multi-objetivo.

    Args:
        wp_vector: Vector de trabajo a modificar
        **kwargs: Parámetros adicionales
            - grupo_info: Información del grupo turístico
            - pois_df: DataFrame de puntos de interés
            - travel_times_df: DataFrame de tiempos de viaje
            - data: Datos completos del problema
            - tipo_turista: Tipo de turista
            - maximize_objectives_list: Lista de objetivos a maximizar
            - origen: Punto de origen (por defecto '1')

    Returns:
        Vector de trabajo modificado
    """
    # Extraer parámetros de kwargs
    grupo_info = kwargs.get('grupo_info')
    pois_df = kwargs.get('pois_df')
    travel_times_df = kwargs.get('travel_times_df')
    data = kwargs.get('data')
    tipo_turista = kwargs.get('tipo_turista')
    maximize_objectives_list = kwargs.get('maximize_objectives_list')
    origen = kwargs.get('origen', '1')

    # Necesitamos también is_dominated y decodificar_wp
    # Asumimos que están disponibles en el scope global

    modified_wp_vector = wp_vector.copy()
    num_elements = len(modified_wp_vector)
    best_vector = wp_vector.copy()
    best_feasible = False
    best_objectives = None

    # Número de vecinos a explorar
    num_neighbors = min(5, num_elements)

    # Generar y evaluar vecinos
    for _ in range(num_neighbors):
        # Crear vecino con pequeña modificación
        neighbor = modified_wp_vector.copy()

        # Modificar un elemento aleatorio
        idx = np.random.randint(num_elements)
        neighbor[idx] = np.random.rand()  # Nuevo valor aleatorio

        # Decodificar y evaluar
        ruta, is_feasible = decodificar_wp(
            wp_vector=neighbor,
            pois_df=pois_df,
            travel_times_df=travel_times_df,
            grupo_info=grupo_info,
            origen=origen
        )

        if is_feasible:
            objectives = evaluar_funciones_objetivo(ruta, data, tipo_turista)

            # Si es el primer vecino factible o es mejor que el mejor actual
            if not best_feasible or is_better_solution(objectives, best_objectives, maximize_objectives_list):
                best_vector = neighbor.copy()
                best_feasible = True
                best_objectives = objectives

    return best_vector

def is_better_solution(new_obj, best_obj, maximize_objectives_list):
    """
    Determina si una solución es mejor que otra usando dominancia de Pareto simplificada.
    Devuelve True si new_obj domina a best_obj o si best_obj es None.
    """
    if best_obj is None:
        return True

    # Convertir objetivos a vectores
    new_vector = [
        new_obj.get('preferencia_total', 0),
        new_obj.get('costo_total', 0),
        new_obj.get('co2_total', 0),
        new_obj.get('sust_ambiental', 0) + new_obj.get('sust_economica', 0) + new_obj.get('sust_social', 0),
        new_obj.get('riesgo_total', 0)
    ]

    best_vector = [
        best_obj.get('preferencia_total', 0),
        best_obj.get('costo_total', 0),
        best_obj.get('co2_total', 0),
        best_obj.get('sust_ambiental', 0) + best_obj.get('sust_economica', 0) + best_obj.get('sust_social', 0),
        best_obj.get('riesgo_total', 0)
    ]

    return is_dominated(best_vector, new_vector, maximize_objectives_list)

def ib_diversity_mutation(wp_vector, **kwargs):
    """
    Intelligence Box específico para fomentar diversidad.
    Realiza mutaciones más agresivas para explorar nuevas áreas del espacio de búsqueda
    """
    modified_wp = wp_vector.copy()

    # Probabilidad de mutación más alta que en operadores normales
    mutation_probability = 0.3  # 30% de probabilidad de mutar cada elemento

    # Extraer datos necesarios
    grupo_info = kwargs.get('grupo_info')
    pois_df = kwargs.get('pois_df')
    current_pareto_front = kwargs.get('current_pareto_front', [])

    # Obtener todos los POIs disponibles
    all_pois = list(pois_df.index)

    # Obtener POIs compatibles con preferencias
    compatible_pois = [poi for poi in all_pois
                     if is_compatible_with_preferences(poi, grupo_info, pois_df)]

    # POIs que no están en soluciones del frente de Pareto
    pareto_pois = set()
    for pareto_solution in current_pareto_front:
        if isinstance(pareto_solution, (list, np.ndarray)):  # Si es iterable
            pareto_pois.update(pareto_solution)

    # POIs novedosos (no usados en el frente de Pareto)
    novel_pois = [poi for poi in compatible_pois if poi not in pareto_pois]

    # Preservar origen
    origen = modified_wp[0]

    # Para cada posición (excepto el origen), decidir si mutar
    for i in range(1, len(modified_wp)):
        if random.random() < mutation_probability:
            # 50% probabilidad de usar un POI novedoso, 50% cualquier compatible
            if novel_pois and random.random() < 0.5:
                candidate_pois = novel_pois
            else:
                candidate_pois = compatible_pois

            # Seleccionar un POI que no esté ya en la solución
            available_pois = [poi for poi in candidate_pois if poi not in modified_wp]

            if available_pois:
                modified_wp[i] = random.choice(available_pois)

    # Asegurar que el origen se mantenga
    modified_wp[0] = origen

    return modified_wp