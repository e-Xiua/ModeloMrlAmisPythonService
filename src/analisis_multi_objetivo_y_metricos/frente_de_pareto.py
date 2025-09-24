import numpy as np

from estado_y_recompensa_rl.definir_comportamiento import is_dominated


def find_pareto_front(objectives_np, maximize_objectives_list):
    """
    Identifica el frente de Pareto a partir de un conjunto de vectores objetivo utilizando comparaciones por pares.

    Argumentos:
        objectives_np (np.ndarray): Una matriz numpy de vectores objetivo (forma: num_solutions, num_objectives).
                                    Cada fila es un valor objetivo de una solución.
        maximize_objectives_list (lista): Una lista de valores booleanos que indican la dirección de optimización
                                         para cada objetivo (True para maximizar, False para minimizar).

    Devuelve:
        tupla: Una tupla que contiene:
            - pareto_front_objectives (np.ndarray): Vectores objetivo de las soluciones del frente de Pareto.
            - pareto_front_indices (np.ndarray): Índices de las soluciones del frente de Pareto en la matriz original objectives_np.
    """
    if objectives_np.shape[0] == 0:
        return np.array([]), np.array([])  # Devuelve vacío si no hay soluciones

    num_solutions = objectives_np.shape[0]
    # Suponer que todas las soluciones son inicialmente no dominadas
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
                # Aquí no hay descanso, ya que seguimos comprobando i con otras soluciones

    # El frente de Pareto está formado por soluciones que no están dominadas por ninguna otra solución
    pareto_front_indices = np.where(~is_dominated_flag)[0]
    pareto_front_objectives = objectives_np[pareto_front_indices]

    return pareto_front_objectives, pareto_front_indices