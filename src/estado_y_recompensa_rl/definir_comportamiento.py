# Esta sección define cómo el agente RL percibe el entorno (el estado)
# y cómo evalúa el resultado de sus acciones (la recompensa).

import numpy as np

# --- Dependencia: is_dominated function ---
# Es probable que esta función ya se encuentre en su cuaderno (en la sección Cálculo de Pareto).
# Inclúyala aquí o asegúrese de que esté definida antes de llamar a get_state.
# La función get_state la necesita para determinar el estado de dominancia.
def is_dominated(point1, point2, maximize_objectives):
    """Comprueba si el punto 1 está dominado por el punto 2 en un sentido multiobjetivo."""
    # Asegurarse de que objectives y maximize_objectives tengan la misma longitud.
    if len(point1) != len(point2) or len(point1) != len(maximize_objectives):
        raise ValueError("Los puntos objetivos y la lista maximize_objectives deben tener la misma longitud.")

    is_better_in_at_least_one = False
    for i in range(len(point1)):
        if maximize_objectives[i]: # Objetivo a maximizar
            if point2[i] > point1[i]:
                is_better_in_at_least_one = True
            elif point2[i] < point1[i]:
                return False # El punto 1 es peor en un objetivo de maximización.
        else: # Objetivo a minimizar
            if point2[i] < point1[i]:
                is_better_in_at_least_one = True
            elif point2[i] > point1[i]:
                return False # El punto 1 es peor en un objetivo de minimización.

    return is_better_in_at_least_one # El punto 1 es dominado si el punto 2 es mejor en al menos un aspecto y no es peor en ninguno.


# --- Función de representación de estado ---
# Supongamos que maximize_objectives_list se ha definido anteriormente en su cuaderno.
# Debe especificar cuáles de los 5 objetivos (en el orden utilizado para Pareto/métricas) se maximizan.
# Ejemplo: maximize_objectives_list = [True, False, False, True, False]

def get_state(wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history=None):
    """
    Define el estado para el agente RL basado en factibilidad de la solución y su relación con el frente de Pareto.

    Estados posibles (0-6):
    0: Infactible y dominada por el frente de Pareto actual
    1: Infactible pero no dominada por el frente de Pareto actual
    2: Factible pero dominada por el frente de Pareto actual
    3: Factible, no dominada, pero no mejora significativamente ningún objetivo
    4: Factible, no dominada, y mejora significativamente al menos un objetivo
    5: Factible, no dominada, y mejora significativamente múltiples objetivos
    6: Estancamiento detectado (hipervolumen sin mejora significativa)

    Args:
        wp_result (dict): Resultado del Work Package a evaluar
        current_pareto_front_objectives (np.ndarray): Objetivos del frente de Pareto actual
        maximize_objectives_list (list): Lista de booleanos indicando qué objetivos maximizar
        hypervolume_history (list, optional): Historial de valores de hipervolumen para detectar estancamiento

    Returns:
        int: Estado discreto (0-6)
    """
    # Verificar factibilidad
    is_feasible = wp_result.get('is_feasible', False)
    objectives_dict = wp_result.get('objetivos', None)

    # Estado predeterminado para casos con objetivos faltantes
    if objectives_dict is None:
        return 0

    # Convertir diccionario de objetivos a vector
    try:
        combined_sustainability = (objectives_dict.get('sust_ambiental', 0) +
                                  objectives_dict.get('sust_economica', 0) +
                                  objectives_dict.get('sust_social', 0))

        objective_vector = [
            objectives_dict.get('preferencia_total', 0),
            objectives_dict.get('costo_total', 0),
            objectives_dict.get('co2_total', 0),
            combined_sustainability,
            objectives_dict.get('riesgo_total', 0)
        ]

        # Verificar longitud del vector
        if len(objective_vector) != len(maximize_objectives_list):
            print(f"Advertencia: Longitud del vector de objetivos ({len(objective_vector)}) "
                  f"no coincide con maximize_objectives_list ({len(maximize_objectives_list)})")
            return 0

    except Exception as e:
        print(f"Error al extraer objetivos: {e}")
        return 0

    # Detectar estancamiento basado en el historial de hipervolumen (estado 6)
    if hypervolume_history is not None and len(hypervolume_history) >= 10:
        # Obtener los últimos 10 valores de hipervolumen
        recent_hv = hypervolume_history[-10:]
        # Calcular la variación relativa
        hv_variation = (max(recent_hv) - min(recent_hv)) / max(0.0001, min(recent_hv))

        # Si hay muy poca variación y la solución es factible y no dominada, consideramos estancamiento
        if hv_variation < 0.001 and is_feasible:
            # Determinar si está dominada por el frente de Pareto actual
            is_dominated_by_pareto = False
            if current_pareto_front_objectives is not None and len(current_pareto_front_objectives) > 0:
                for pareto_obj_vector in current_pareto_front_objectives:
                    if is_dominated(objective_vector, pareto_obj_vector, maximize_objectives_list):
                        is_dominated_by_pareto = True
                        break

            # Si no está dominada, consideramos estancamiento
            if not is_dominated_by_pareto:
                return 6  # Estado de estancamiento

    # Determinar si está dominada por el frente de Pareto actual
    is_dominated_by_pareto = False
    if current_pareto_front_objectives is not None and len(current_pareto_front_objectives) > 0:
        for pareto_obj_vector in current_pareto_front_objectives:
            if is_dominated(objective_vector, pareto_obj_vector, maximize_objectives_list):
                is_dominated_by_pareto = True
                break

    # Determinar si mejora significativamente algún objetivo del frente de Pareto
    significant_improvements = 0
    if not is_dominated_by_pareto and current_pareto_front_objectives is not None and len(current_pareto_front_objectives) > 0:
        # Calcular valores promedio para cada objetivo en el frente de Pareto
        pareto_avg = np.mean(current_pareto_front_objectives, axis=0)

        # Contar mejoras significativas (>5%)
        for i, maximize in enumerate(maximize_objectives_list):
            if maximize:  # Objetivos a maximizar
                if objective_vector[i] > 1.05 * pareto_avg[i]:
                    significant_improvements += 1
            else:  # Objetivos a minimizar
                if objective_vector[i] < 0.95 * pareto_avg[i]:
                    significant_improvements += 1

    # Determinar estado según factibilidad, dominancia y mejoras
    if not is_feasible:
        return 0 if is_dominated_by_pareto else 1
    else:  # Es factible
        if is_dominated_by_pareto:
            return 2
        else:  # No dominada
            if significant_improvements == 0:
                return 3
            elif significant_improvements == 1:
                return 4
            else:  # Múltiples mejoras
                return 5

def calculate_reward(original_wp_result, modified_wp_result,
                    current_pareto_front_objectives, maximize_objectives_list):
    """
    Calcula la recompensa por aplicar un operador, incluyendo bonificación por diversidad.

    Args:
        original_wp_result (dict): Resultado del WP original
        modified_wp_result (dict): Resultado del WP modificado
        current_pareto_front_objectives (list/ndarray): Objetivos del frente de Pareto actual
        maximize_objectives_list (list): Lista de booleanos indicando qué objetivos maximizar

    Returns:
        float: Valor de recompensa
    """
    # Verificar factibilidad
    original_feasible = original_wp_result.get('is_feasible', False)
    modified_feasible = modified_wp_result.get('is_feasible', False)

    # Alta recompensa por convertir una solución infactible en factible
    if not original_feasible and modified_feasible:
        return 10.0

    # Penalización por convertir una solución factible en infactible
    if original_feasible and not modified_feasible:
        return -5.0

    # Si ambas son infactibles
    if not original_feasible and not modified_feasible:
        return -1.0

    # Extraer objetivos
    original_objectives = original_wp_result.get('objetivos', {})
    modified_objectives = modified_wp_result.get('objetivos', {})

    if not original_objectives or not modified_objectives:
        return 0.0

    # Convertir diccionarios de objetivos a vectores
    try:
        original_obj_vector = [
            original_objectives.get('preferencia_total', 0),
            original_objectives.get('costo_total', 0),
            original_objectives.get('co2_total', 0),
            original_objectives.get('sust_ambiental', 0) +
            original_objectives.get('sust_economica', 0) +
            original_objectives.get('sust_social', 0),
            original_objectives.get('riesgo_total', 0)
        ]

        modified_obj_vector = [
            modified_objectives.get('preferencia_total', 0),
            modified_objectives.get('costo_total', 0),
            modified_objectives.get('co2_total', 0),
            modified_objectives.get('sust_ambiental', 0) +
            modified_objectives.get('sust_economica', 0) +
            modified_objectives.get('sust_social', 0),
            modified_objectives.get('riesgo_total', 0)
        ]
    except Exception as e:
        print(f"Error al extraer vectores de objetivos: {e}")
        return 0.0

    # --- RECOMPENSA BASE (por mejora en objetivos) ---
    base_reward = 0.0

    # Calcular mejora en cada objetivo
    for i, value in enumerate(original_obj_vector):
        if i < len(maximize_objectives_list):
            # Para objetivos a maximizar, mejor es mayor
            if maximize_objectives_list[i]:
                improvement = modified_obj_vector[i] - original_obj_vector[i]
                # Normalizar mejora usando el valor original
                if original_obj_vector[i] != 0:
                    rel_improvement = improvement / abs(original_obj_vector[i])
                else:
                    rel_improvement = improvement if improvement > 0 else 0

                base_reward += min(rel_improvement, 1.0)  # Limitar mejoras extremas

            # Para objetivos a minimizar, mejor es menor
            else:
                improvement = original_obj_vector[i] - modified_obj_vector[i]
                # Normalizar mejora usando el valor original
                if original_obj_vector[i] != 0:
                    rel_improvement = improvement / abs(original_obj_vector[i])
                else:
                    rel_improvement = improvement if improvement > 0 else 0

                base_reward += min(rel_improvement, 1.0)  # Limitar mejoras extremas

    # Normalizar por número de objetivos
    base_reward = base_reward / len(maximize_objectives_list)

    # --- RECOMPENSA POR DIVERSIDAD ---
    diversity_reward = 0.0

    # Solo calcular recompensa por diversidad si la solución es factible y hay un frente de Pareto
    has_pareto_front = False
    if current_pareto_front_objectives is not None:
        if isinstance(current_pareto_front_objectives, list):
            has_pareto_front = len(current_pareto_front_objectives) > 0
        elif isinstance(current_pareto_front_objectives, np.ndarray):
            has_pareto_front = current_pareto_front_objectives.size > 0

    if modified_feasible and has_pareto_front:
        # Verificar si la solución modificada está dominada por el frente de Pareto actual
        is_dominated_by_pareto = False
        for pareto_obj in current_pareto_front_objectives:
            if isinstance(pareto_obj, (list, np.ndarray)):
                if is_dominated(modified_obj_vector, pareto_obj, maximize_objectives_list):
                    is_dominated_by_pareto = True
                    break

        # Si no está dominada, calcular su contribución a la diversidad
        if not is_dominated_by_pareto:
            # Calcular distancia mínima al frente de Pareto actual
            min_distance = float('inf')

            for pareto_obj in current_pareto_front_objectives:
                if isinstance(pareto_obj, (list, np.ndarray)):
                    # Calcular distancia normalizada
                    distance = 0.0
                    for i in range(min(len(modified_obj_vector), len(pareto_obj))):
                        # Normalizar valores para cada objetivo
                        obj_range = get_objective_range(i, current_pareto_front_objectives)
                        if obj_range > 0:
                            norm_diff = abs(modified_obj_vector[i] - pareto_obj[i]) / obj_range
                            distance += norm_diff ** 2

                    distance = np.sqrt(distance)
                    min_distance = min(min_distance, distance)

            # Recompensa por diversidad: mayor distancia = mayor recompensa
            if min_distance < float('inf'):
                # Convertir distancia a recompensa:
                # - Pequeña distancia (solución similar): poca recompensa
                # - Gran distancia (solución diversa): alta recompensa
                diversity_reward = min(min_distance * 2.0, 2.0)  # Limitar a 2.0 máximo

    # --- RECOMPENSA FINAL ---
    # Combinar recompensa base y recompensa por diversidad
    # Dar más peso a la diversidad cuando haya más soluciones en el frente
    pareto_size = (len(current_pareto_front_objectives) if isinstance(current_pareto_front_objectives, list)
                  else current_pareto_front_objectives.shape[0] if isinstance(current_pareto_front_objectives, np.ndarray) and current_pareto_front_objectives.size > 0
                  else 0)

    diversity_weight = min(0.5, pareto_size / 100) if pareto_size > 0 else 0
    total_reward = (1 - diversity_weight) * base_reward + diversity_weight * diversity_reward

    # Limitar la recompensa para evitar valores extremos
    total_reward = max(min(total_reward, 5.0), -5.0)

    return total_reward

def get_objective_range(obj_index, objectives_list):
    """
    Calcula el rango (max - min) para un objetivo específico en una lista de vectores de objetivos
    """
    # Verificar si objectives_list es None o está vacío
    if objectives_list is None:
        return 1.0  # Valor por defecto si no hay datos

    if isinstance(objectives_list, list):
        if len(objectives_list) == 0:
            return 1.0  # Valor por defecto si la lista está vacía
    elif isinstance(objectives_list, np.ndarray):
        if objectives_list.size == 0:
            return 1.0  # Valor por defecto si el array está vacío

    values = []
    for obj in objectives_list:
        if isinstance(obj, (list, np.ndarray)) and obj_index < len(obj):
            values.append(obj[obj_index])

    if not values:
        return 1.0

    return max(values) - min(values) if max(values) > min(values) else 1.0


# Definir el tamaño del espacio de estados para el agente RL (basado en la función get_state)
# state_space_size = 4 # Basado en los 4 estados posibles devueltos por get_state