import numpy as np
import pandas as pd


def generar_work_packages(q, d, random_state=42):
    """
    Genera q Work Packages (WPs) aleatorios de dimensión d.
    Cada WP es un vector de valores entre 0 y 1.
    """
    np.random.seed(random_state)
    # Generar q WPs aleatorios de dimensión d
    work_packages = np.random.rand(q, d)
    return pd.DataFrame(work_packages, index=[f"WP_{i+1}" for i in range(q)])

def decodificar_wp(wp_vector, pois_df, travel_times_df, grupo_info, origen='1'):
    """
    Decodifica un WP (vector de valores float) en una ruta,
    verificando restricciones y devolviendo la ruta y su factibilidad.

    Args:
        wp_vector: Vector del Work Package (valores 0-1)
        pois_df: DataFrame con información de POIs
        travel_times_df: DataFrame con tiempos de viaje entre POIs
        grupo_info: Diccionario con información del grupo turístico
        origen: ID del POI de origen/hotel

    Returns:
        tuple: (ruta_completa, is_feasible)
    """
    is_feasible = True
    ruta = []
    actual = origen
    hora_actual = 480  # 8:00 AM (Es importante establecer una hora de inicio del tour, de lo contrario, el modelo toma como default 12:00a.m)

    # Verificar parámetros de entrada
    if not isinstance(pois_df, pd.DataFrame) or not isinstance(travel_times_df, pd.DataFrame):
        print("Error: Dataframes de entrada inválidos")
        return [origen, origen], False

    # Verificar que el origen existe en dataframes
    if origen not in travel_times_df.index or origen not in travel_times_df.columns:
        print(f"Error: POI de origen '{origen}' no encontrado en dataframes")
        return [origen, origen], False

    # Obtener restricciones del grupo
    TMax = grupo_info.get('tiempo_disponible', float('inf'))
    min_pois = grupo_info.get('min_pois_per_route', 0)
    max_pois = grupo_info.get('max_pois_per_route', float('inf'))

    # Ordenar POIs según valores del WP
    orden_pois = wp_vector.argsort()
    pois_disponibles = list(pois_df.index.astype(str))

    # Construir ruta según el orden de valores
    for pos in orden_pois:
        if pos < 0 or pos >= len(pois_disponibles):
            continue

        poi_candidato = pois_disponibles[pos]

        # Verificar si el POI es repetido
        if poi_candidato == actual or poi_candidato in ruta:
            continue

        # Verificar tiempos de viaje disponibles
        if actual not in travel_times_df.index or poi_candidato not in travel_times_df.columns:
            continue

        t_traslado = travel_times_df.loc[actual, poi_candidato]

        # Obtener atributos del POI
        try:
            duracion_visita = pois_df.loc[poi_candidato, 'duracion_visita_min']
            horario_apertura = pois_df.loc[poi_candidato, 'horario_apertura']
            horario_cierre = pois_df.loc[poi_candidato, 'horario_cierre']
        except:
            continue

        # Calcular tiempos de llegada y salida
        llegada = hora_actual + t_traslado

        # Esperar si llega antes de apertura
        if llegada < horario_apertura:
            llegada = horario_apertura

        salida = llegada + duracion_visita

        # Verificar si cumple ventana de tiempo
        if salida > horario_cierre:
            continue

        # Aceptar el POI
        ruta.append(poi_candidato)
        hora_actual = salida
        actual = poi_candidato

        # Verificar si ya alcanzamos el máximo de POIs
        if len(ruta) >= max_pois:
            break

    # Verificar restricciones finales
    if len(ruta) < min_pois:
        is_feasible = False

    # Simular la ruta completa para verificar duración total
    if ruta:
        tiempo_total = simular_ruta_completa(ruta, origen, pois_df, travel_times_df)
        if (tiempo_total - 480) > TMax:  # Verificar tiempo máximo
            is_feasible = False
    else:
        is_feasible = min_pois == 0  # Ruta vacía solo factible si min_pois es 0

    # Incluir origen al inicio y final de la ruta
    ruta_completa = [origen] + ruta + [origen] if ruta else [origen, origen]

    return ruta_completa, is_feasible

def simular_ruta_completa(ruta, origen, pois_df, travel_times_df):
    """
    Simula la ruta para calcular el tiempo total incluyendo retorno al origen.
    Devuelve el tiempo final (hora de llegada de vuelta al origen).
    """
    hora = 480  # 8:00 AM
    ubicacion = origen

    # Simular viaje a cada POI
    for poi in ruta:
        # Tiempo de viaje al POI
        if ubicacion in travel_times_df.index and poi in travel_times_df.columns:
            t_viaje = travel_times_df.loc[ubicacion, poi]
        else:
            return float('inf')  # Error en datos de tiempo

        hora += t_viaje

        # Ventana de tiempo y duración
        try:
            apertura = pois_df.loc[poi, 'horario_apertura']
            cierre = pois_df.loc[poi, 'horario_cierre']
            duracion = pois_df.loc[poi, 'duracion_visita_min']
        except:
            return float('inf')  # Error en datos de POI

        # Esperar si llega antes de apertura
        if hora < apertura:
            hora = apertura

        # Verificar factibilidad de ventana temporal
        if hora > cierre:
            return float('inf')

        # Tiempo de visita
        hora += duracion
        ubicacion = poi

    # Tiempo de regreso al origen
    if ubicacion in travel_times_df.index and origen in travel_times_df.columns:
        t_regreso = travel_times_df.loc[ubicacion, origen]
        hora += t_regreso
    else:
        return float('inf')

    return hora