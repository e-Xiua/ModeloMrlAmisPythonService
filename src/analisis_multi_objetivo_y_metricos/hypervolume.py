import numpy as np
from pymoo.indicators.hv import HV



def calculate_hypervolume(objectives_np, reference_point, maximize_objectives_list):
    """
    Calcula el hipervolumen normalizado (rango 0-1) de un conjunto de objetivos.

    Argumentos:
        objectives_np (np.ndarray): Una matriz numpy de vectores objetivo (forma: num_solutions, num_objectives)
        reference_point (np.ndarray): punto de referencia (peores valores para cada objetivo)
        maximize_objectives_list (list): lista de valores booleanos que indican qué objetivos maximizar

    Devuelve:
        float: valor de hipervolumen normalizado entre 0,0 y 1,0
    """
    
    if len(objectives_np) == 0:
        return 0.0  # El conjunto vacío tiene hipervolumen cero

    num_objectives = objectives_np.shape[1]

    # 1. Calcular los límites objetivos para la normalización
    min_bounds = np.min(objectives_np, axis=0)
    max_bounds = np.max(objectives_np, axis=0)

    # Evita división por cero (0)
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

    # 5. Asegúrese de que el punto de referencia sea peor que todas las soluciones
    for i in range(num_objectives):
        max_obj = np.max(objectives_minimization[:, i])
        reference_minimization[i] = max(reference_minimization[i], max_obj * 1.1)

    # 6. Calcular el hipervolumen utilizando pymoo
    try:
        # pymoo HV indicator - ref point must be worse than all points
        hv_indicator = HV(ref_point=reference_minimization)
        hv_value = hv_indicator(objectives_minimization)

        # 7. Normalizar al rango [0,1]
        # Calcular el hipervolumen máximo posible en este espacio normalizado
        max_possible_hv = np.prod(reference_minimization - np.zeros(num_objectives))

        # Devuelve el hipervolumen normalizado
        normalized_hv = hv_value / max_possible_hv if max_possible_hv > 0 else 0.0

        return min(1.0, max(0.0, normalized_hv))  # Asegurar que los valores estén en [0,1]

    except Exception as e:
        print(f"Error calculando hypervolume: {e}")
        return 0.0