def average_ratio_pareto(solutions_np, population_np, maximize_objectives_list):
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

    # Para el ARP estándar, solo necesitamos la proporción de soluciones no dominadas en la población.
    # Pero ya tenemos el conjunto de soluciones no dominadas, por lo que:
    return len(solutions_np) / len(population_np)