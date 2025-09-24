from estado_y_recompensa_rl.definir_comportamiento import is_dominated


def set_coverage(set_a, set_b, maximize_objectives_list):
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