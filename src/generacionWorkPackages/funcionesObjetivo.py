def evaluar_funciones_objetivo(ruta, data, tipo_turista="nacional"):
    """
    Evalúa una ruta en términos de los objetivos de optimización.

    Args:
        ruta (list): Lista de POIs en la ruta (incluyendo origen al inicio y final)
        data (dict): Diccionario con todos los dataframes de datos
        tipo_turista (str): Tipo de turista (nacional o extranjero)

    Returns:
        dict: Diccionario con valores para cada objetivo
    """
    # Acceder a dataframes necesarios
    pois_df = data["pois"]
    costos_df = data["costs"]
    co2_df = data["co2_emission_cost"]
    riesgo_df = data["accident_risk"]
    experiencia_df = data["costos_experiencia"]

    # Determinar columna de costo según tipo de turista
    columna_costo_exp = "costo_nacional_exp_USD" if tipo_turista.lower() == "nacional" else "costo_extranjero_exp_USD"

    # Obtener POIs intermedios (sin origen/destino)
    pois_intermedios = ruta[1:-1]

    # Evaluar preferencia y sostenibilidad (basado en POIs)
    preferencia = pois_df.loc[pois_intermedios, "preferencia"].sum()
    sust_ambiental = pois_df.loc[pois_intermedios, "sust_ambiental"].sum()
    sust_economica = pois_df.loc[pois_intermedios, "sust_economica"].sum()
    sust_social = pois_df.loc[pois_intermedios, "sust_social"].sum()

    # Evaluar costos, CO2 y riesgo (basado en conexiones)
    costo_total = 0
    co2_total = 0
    riesgo_total = 0

    # Sumar costos por trayectos
    for i in range(len(ruta) - 1):
        origen = ruta[i]
        destino = ruta[i + 1]

        costo_total += costos_df.loc[origen, destino]
        co2_total += co2_df.loc[origen, destino]
        riesgo_total += riesgo_df.loc[origen, destino]

    # Agregar costo de experiencia en cada POI
    for poi in pois_intermedios:
        costo_total += experiencia_df.loc[poi, columna_costo_exp]

    return {
        "preferencia_total": preferencia,
        "costo_total": costo_total,
        "co2_total": co2_total,
        "sust_ambiental": sust_ambiental,
        "sust_economica": sust_economica,
        "sust_social": sust_social,
        "riesgo_total": riesgo_total
    }

def is_compatible_with_preferences(poi_id, grupo_info, pois_df):
    """
    Verifica si un POI es compatible con las preferencias del grupo de turistas.

    Args:
        poi_id (str): ID del POI a evaluar
        grupo_info (dict): Información del grupo con sus preferencias
        pois_df (pd.DataFrame): DataFrame con información de los POIs

    Returns:
        bool: True si el POI es compatible, False en caso contrario
    """
    if poi_id not in pois_df.index:
        return False

    poi = pois_df.loc[poi_id]

    # Verificar si el tipo de POI está en las preferencias del grupo
    if 'preferencias_tipo_actividad' in grupo_info and poi['tipo'] in grupo_info['preferencias_tipo_actividad']:
        return True

    # Si llegamos aquí, el POI no es compatible
    return False