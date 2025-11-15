"""Utilidades para transformar las solicitudes gRPC en las dataclasses de dominio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional
import math
import logging

import numpy as np
import pandas as pd

from runmodel.models.poi import POI
from runmodel.models.tourist_group import TouristGroup

logger = logging.getLogger(__name__)

# ---- Constantes y utilidades internas ------------------------------------------------------

CATEGORY_TO_POI_TYPE = {
    "yoga": "yoga",
    "meditación": "meditación",
    "meditacion": "meditación",
    "spa": "spa",
    "masajes": "masajes",
    "senderismo": "senderismo",
    "aguas termales": "aguas termales",
    "nutrición saludable": "nutrición saludable",
    "nutricion saludable": "nutrición saludable",
    "fitness": "fitness",
    "mindfulness": "mindfulness",
    "retiros espirituales": "retiros espirituales",
    "aromaterapia": "aromaterapia",
    "pilates": "pilates",
    "tai chi": "tai chi",
    "naturopatía": "naturopatía",
    "naturopatia": "naturopatía",
    "terapias holísticas": "terapias holísticas",
    "terapias holisticas": "terapias holísticas",
}
DEFAULT_POI_TYPE = "bienestar_general"

BASE_CATEGORY_VECTOR = {
    "yoga": 60.0,
    "meditación": 60.0,
    "spa": 60.0,
    "masajes": 60.0,
    "senderismo": 60.0,
    "aguas termales": 60.0,
    "nutrición saludable": 60.0,
    "fitness": 60.0,
    "mindfulness": 60.0,
    "retiros espirituales": 60.0,
    "aromaterapia": 60.0,
    "pilates": 60.0,
    "tai chi": 60.0,
    "naturopatía": 60.0,
    "terapias holísticas": 60.0,
    "bienestar_general": 50.0,
}


@dataclass
class DomainPayload:
    """Datos derivados que consume el worker MRL-AMIS."""

    pois: List[POI]
    tourist_group: TouristGroup
    pois_dataframe: pd.DataFrame
    distance_matrix: pd.DataFrame
    travel_time_matrix: pd.DataFrame
    co2_matrix: pd.DataFrame
    group_preferences_matrix: pd.DataFrame
    internal_to_external_ids: Dict[str, str]
    raw_request: Dict[str, Any]


# ---- API pública ---------------------------------------------------------------------------


def build_domain_payload(
    *,
    job_data: Dict[str, Any],
    request_data: Dict[str, Any],
    default_group_id: str,
) -> DomainPayload:
    """Construye toda la estructura necesaria para el worker."""
    
    # LOG 6: Datos de entrada para el mapeo de dominio
    logger.info(f"PASO 6: Iniciando `build_domain_payload` para trabajo {job_data.get('route_id')}")
    logger.info(f"  -> `request_data` de entrada: {request_data}")

    # Mapear POIs
    mapped_pois = [_map_poi_dict(p, index) for index, p in enumerate(request_data.get("pois", []), start=1)]
    pois = [poi for poi, _ in mapped_pois]
    id_lookup = {poi.id: external_id for poi, external_id in mapped_pois}
    
    # Crear tourist group con manejo robusto de start_location
    tourist_group = _map_tourist_group(
        job_data=job_data,
        request_data=request_data,
        pois=pois,
        default_group_id=default_group_id,
    )
    
    # LOG 8: TouristGroup y POIs mapeados
    logger.info(f"PASO 8: Mapeo a dataclasses completado para trabajo {job_data.get('route_id')}")
    logger.info(f"  -> POIs mapeados: {len(pois)}")
    logger.info(f"  -> TouristGroup origen: {tourist_group.origen}, destino: {tourist_group.destino}")

    # Generar estructuras de datos
    pois_df = _pois_to_dataframe(pois)
    distance_matrix, travel_time_matrix = _build_distance_and_time_matrices(pois)
    co2_matrix = _build_co2_matrix(pois, travel_time_matrix)
    preferences_matrix = _build_group_preferences_matrix(tourist_group, pois)

    return DomainPayload(
        pois=pois,
        tourist_group=tourist_group,
        pois_dataframe=pois_df,
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
        co2_matrix=co2_matrix,
        group_preferences_matrix=preferences_matrix,
        internal_to_external_ids=id_lookup,
        raw_request=request_data,
    )


# ---- Implementación interna ----------------------------------------------------------------


def _map_poi_dict(poi_dict: Dict[str, Any], index: int) -> Tuple[POI, str]:
    """Transforma un diccionario proveniente de gRPC en una instancia de `POI`."""

    external_id = str(poi_dict.get("id") or poi_dict.get("poi_id") or f"poi_{index}")
    poi_id = str(index)  # ID interno siempre numérico

    name = str(poi_dict.get("name") or poi_dict.get("nombre") or f"POI {external_id}")

    # Manejo robusto de coordenadas
    latitude = _coerce_float(poi_dict.get("latitude") or poi_dict.get("lat") or poi_dict.get("latitud"))
    longitude = _coerce_float(poi_dict.get("longitude") or poi_dict.get("lng") or poi_dict.get("longitud"))
    
    # Si no hay coordenadas, usar valores por defecto basados en índice
    if latitude == 0.0 and longitude == 0.0:
        # Generar coordenadas ficticias pero consistentes
        latitude = 40.0 + (index * 0.01)
        longitude = -3.0 + (index * 0.01)

    visit_duration = max(int(poi_dict.get("visit_duration") or poi_dict.get("duracion_visita") or 60), 15)
    duration_hours = max(visit_duration / 60.0, 0.25)

    cost = _coerce_float(poi_dict.get("cost") or poi_dict.get("precio") or poi_dict.get("costo") or 0.0)
    stay_cost_per_hour = cost / duration_hours if cost else 0.0

    rating = _coerce_float(poi_dict.get("rating") or poi_dict.get("calificacion") or 3.5)
    preference_score = max(0.0, min(100.0, rating * 20 if rating <= 5 else rating))

    category = str(poi_dict.get("subcategory") or poi_dict.get("category") or "").lower()
    poi_type = CATEGORY_TO_POI_TYPE.get(category, DEFAULT_POI_TYPE)

    opening_hours = poi_dict.get("opening_hours")
    open_time, close_time = _parse_opening_hours(opening_hours, default_range=(800, 1800))

    sust_ambiental = _coerce_float(poi_dict.get("environment_score") or preference_score * 0.8)
    sust_economica = _coerce_float(poi_dict.get("economic_score") or preference_score * 0.75)
    sust_social = _coerce_float(poi_dict.get("social_score") or preference_score * 0.7)

    riesgo_salud = max(5.0, 100.0 - sust_social)
    riesgo_accidente = max(5.0, 100.0 - sust_ambiental)

    seguridad = float(poi_dict.get("safety_score") or 80.0)
    capacidad_maxima = int(poi_dict.get("capacity") or poi_dict.get("capacidad_maxima") or 150)

    poi_instance = POI(
        id=poi_id,
        name=name,
        position=(latitude, longitude),
        stay_cost_per_hour=stay_cost_per_hour,
        entry_cost=max(0.0, cost * 0.25),
        co2_per_hour=float(poi_dict.get("co2_per_hour") or 0.35),
        preferencia=preference_score,
        sust_ambiental=sust_ambiental,
        sust_economica=sust_economica,
        sust_social=sust_social,
        riesgo_salud=riesgo_salud,
        riesgo_accidente=riesgo_accidente,
        horario_apertura=open_time,
        horario_cierre=close_time,
        duracion_visita_min=visit_duration,
        seguridad_poi=seguridad,
        capacidad_maxima=capacidad_maxima,
        tipo_poi=poi_type,
    )

    return poi_instance, external_id


def _map_tourist_group(
    *,
    job_data: Dict[str, Any],
    request_data: Dict[str, Any],
    pois: List[POI],
    default_group_id: str,
) -> TouristGroup:
    """Genera una instancia de `TouristGroup` a partir de la solicitud."""

    preferences = request_data.get("preferences") or {}
    constraints = request_data.get("constraints") or {}

    preferred_categories = _ensure_iterable(preferences.get("preferred_categories"))
    avoid_categories = set(_ensure_iterable(preferences.get("avoid_categories")))
    
    logger.info(f"PASO 7: Mapeando TouristGroup con constraints: {constraints}")

    category_vector = BASE_CATEGORY_VECTOR.copy()
    for category in preferred_categories:
        normalized = str(category).lower()
        mapped = CATEGORY_TO_POI_TYPE.get(normalized, normalized)
        category_vector[mapped] = max(category_vector.get(mapped, 50.0), 80.0)

    for category in avoid_categories:
        normalized = str(category).lower()
        mapped = CATEGORY_TO_POI_TYPE.get(normalized, normalized)
        category_vector[mapped] = min(category_vector.get(mapped, 50.0), 30.0)

    max_total_cost = _coerce_float(preferences.get("max_total_cost") or _estimate_total_cost(pois))
    time_available = int(preferences.get("max_total_time") or _estimate_total_time(pois))

    if time_available <= 0:
        time_available = 720

    group_size = int(preferences.get("group_size") or max(1, math.ceil(len(pois) * 1.5)))
    tourist_type = str(preferences.get("tourist_type") or "custom")

    # Manejo robusto de origen y destino
    origen = _determine_origin(constraints, pois)
    destino = _determine_destination(constraints, pois, origen)

    sustainability_threshold = _coerce_float(preferences.get("sustainability_min") or 60.0)

    conciencia_ambiental = max(
        sustainability_threshold,
        sustainability_threshold + (5 if "naturaleza" in preferred_categories else 0),
    )

    tolerancia_riesgo = 40.0 if preferences.get("accessibility_required") else 60.0

    presupuesto_min = max_total_cost * 0.5
    limite_distancia = float(preferences.get("max_distance_km") or 80.0)

    return TouristGroup(
        grupo_id=str(preferences.get("group_id") or default_group_id),
        tipo_turista=tourist_type,
        tamano_grupo=group_size,
        origen=origen,
        destino=destino,
        presupuesto_max_usd=max_total_cost,
        presupuesto_min_usd=presupuesto_min,
        sostenibilidad_min=sustainability_threshold,
        limite_max_dist_km=limite_distancia,
        tiempo_disponible_min=time_available,
        preferencias_tipos=category_vector,
        nivel_aventura=float(preferences.get("adventure_level") or category_vector.get("aventura", 50.0)),
        sensibilidad_costo=float(preferences.get("cost_sensitivity") or 50.0),
        conciencia_ambiental=conciencia_ambiental,
        tolerancia_riesgo=tolerancia_riesgo,
    )


def _determine_origin(constraints: Dict[str, Any], pois: List[POI]) -> str:
    """Determina el POI de origen de manera robusta."""
    start_location = constraints.get("start_location")
    
    # Si hay start_location con coordenadas, encontrar el POI más cercano
    if start_location and start_location.get("latitude") is not None and start_location.get("longitude") is not None:
        logger.info(f"  -> `_determine_origin`: Encontrado `start_location` explícito: {start_location}")
        closest_poi = _find_closest_poi_to_location(pois, start_location)
        if closest_poi:
            logger.info(f"  -> `_determine_origin`: POI más cercano encontrado: ID {closest_poi.id}")
            return closest_poi.id
    
    # Si no, usar el primer POI disponible
    if pois:
        logger.info(f"  -> `_determine_origin`: Usando primer POI como fallback: ID {pois[0].id}")
        return pois[0].id
    
    # Fallback
    logger.warning("  -> `_determine_origin`: No hay `start_location` ni POIs, usando fallback '1'")
    return "1"


def _determine_destination(constraints: Dict[str, Any], pois: List[POI], default_origin: str) -> str:
    """Determina el POI de destino de manera robusta."""
    end_location = constraints.get("end_location")
    
    # Si hay end_location con coordenadas, encontrar el POI más cercano
    if end_location and end_location.get("latitude") is not None and end_location.get("longitude") is not None:
        logger.info(f"  -> `_determine_destination`: Encontrado `end_location` explícito: {end_location}")
        closest_poi = _find_closest_poi_to_location(pois, end_location)
        if closest_poi:
            logger.info(f"  -> `_determine_destination`: POI más cercano encontrado: ID {closest_poi.id}")
            return closest_poi.id
    
    # Si no, usar el último POI disponible (diferente al origen si es posible)
    if pois:
        if len(pois) > 1 and pois[-1].id != default_origin:
            return pois[-1].id
        elif len(pois) > 2:
            return pois[-2].id
    
    # Fallback: mismo que origen
    return default_origin


def _pois_to_dataframe(pois: List[POI]) -> pd.DataFrame:
    """Convierte la lista de POIs en el DataFrame que espera el worker."""

    rows: List[Dict[str, Any]] = []
    for poi in pois:
        rows.append(
            {
                "poi_id": poi.id,
                "nombre": poi.name,
                "latitud": poi.position[0],
                "longitud": poi.position[1],
                "categoria": poi.tipo_poi,
                "costo": poi.entry_cost + poi.stay_cost_per_hour * (poi.duracion_visita_min / 60.0),
                "calificacion": poi.preferencia / 20.0 if poi.preferencia <= 100 else poi.preferencia,
                "preferencia": poi.preferencia,
                "duracion_visita_min": poi.duracion_visita_min,
                "horario_apertura": poi.horario_apertura,
                "horario_cierre": poi.horario_cierre,
                "sust_ambiental": poi.sust_ambiental,
                "sust_social": poi.sust_social,
                "sust_economica": poi.sust_economica,
                "riesgo_salud": poi.riesgo_salud,
                "riesgo_accidente": poi.riesgo_accidente,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "poi_id",
                "nombre",
                "latitud",
                "longitud",
                "categoria",
                "costo",
                "calificacion",
                "preferencia",
                "duracion_visita_min",
                "horario_apertura",
                "horario_cierre",
                "sust_ambiental",
                "sust_social",
                "sust_economica",
                "riesgo_salud",
                "riesgo_accidente",
            ]
        ).set_index("poi_id")

    # CRITICAL FIX: Set poi_id as index (MRL-AMIS expects POI IDs as index)
    df = pd.DataFrame(rows)
    df = df.set_index("poi_id")
    return df


def _build_distance_and_time_matrices(pois: List[POI]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Genera matrices de distancias y tiempos de viaje basadas en coordenadas."""

    num_pois = len(pois)
    if num_pois == 0:
        empty_df = pd.DataFrame()
        return empty_df, empty_df

    distances = np.zeros((num_pois, num_pois))
    travel_times = np.zeros((num_pois, num_pois))

    for i, origin in enumerate(pois):
        for j, target in enumerate(pois):
            if i == j:
                continue
            distance = _haversine(origin.position, target.position)
            distances[i, j] = distance
            travel_times[i, j] = distance / 40 * 60  # 40 km/h -> minutos

    labels = [poi.id for poi in pois]
    return (
        pd.DataFrame(distances, index=labels, columns=labels),
        pd.DataFrame(travel_times, index=labels, columns=labels),
    )


def _build_co2_matrix(pois: List[POI], travel_time_matrix: pd.DataFrame) -> pd.DataFrame:
    """Genera matriz de emisiones de CO2 basada en tiempos de viaje y características de POIs."""
    
    num_pois = len(pois)
    if num_pois == 0:
        return pd.DataFrame()
    
    co2_emissions = np.zeros((num_pois, num_pois))
    labels = [poi.id for poi in pois]
    
    for i, origin_poi in enumerate(pois):
        for j, target_poi in enumerate(pois):
            if i == j:
                continue
            
            # CO2 por transporte: tiempo de viaje * factor de emisión por minuto
            travel_time_minutes = travel_time_matrix.iloc[i, j]
            transport_co2 = travel_time_minutes * 0.02  # ~0.02 kg CO2 por minuto de viaje
            
            # CO2 por estadía en el POI destino (usar co2_per_hour del POI)
            stay_time_hours = target_poi.duracion_visita_min / 60.0
            stay_co2 = stay_time_hours * target_poi.co2_per_hour
            
            # Total de emisiones para el viaje desde origin_poi hasta target_poi
            co2_emissions[i, j] = transport_co2 + stay_co2
    
    return pd.DataFrame(co2_emissions, index=labels, columns=labels)


def _build_group_preferences_matrix(group: TouristGroup, pois: List[POI]) -> pd.DataFrame:
    """Construye matriz de preferencias del grupo turístico como DataFrame.
    
    Returns:
        DataFrame donde:
        - Índice (filas) = grupo_id
        - Columnas = poi_id (como strings: '1', '2', '3', etc.)
        - Valores = preference_score (0-100)
    """
    
    # Crear diccionario de preferencias para este grupo
    poi_preferences = {}
    
    for poi in pois:
        # Calcular score de preferencia basado en tipo de POI y preferencias del grupo
        poi_type = poi.tipo_poi
        base_score = group.preferencias_tipos.get(poi_type, 50.0)
        
        # Ajustar por nivel de aventura si es POI de aventura
        if poi_type == 'aventura':
            base_score *= (1.0 + group.nivel_aventura / 100.0 * 0.5)
        
        # Ajustar por conciencia ambiental si es naturaleza
        if poi_type == 'naturaleza':
            base_score *= (1.0 + group.conciencia_ambiental / 100.0 * 0.3)
        
        # Ajustar por preferencia intrínseca del POI
        poi_preference_factor = poi.preferencia / 100.0  # Normalizar a 0-1
        adjusted_score = base_score * (0.7 + 0.3 * poi_preference_factor)
        
        # Limitar a rango válido
        final_score = max(0.0, min(100.0, adjusted_score))
        
        # Usar ID del POI como columna (string)
        poi_preferences[poi.id] = final_score
    
    # Crear DataFrame con grupo_id como índice (fila) y poi_ids como columnas
    df = pd.DataFrame([poi_preferences], index=[group.grupo_id])
    
    logger.info(f"Matriz de preferencias (DataFrame) construida:")
    logger.info(f"  -> Shape: {df.shape} (grupos x POIs)")
    logger.info(f"  -> Índice (grupos): {df.index.tolist()}")
    logger.info(f"  -> Columnas (POIs): {df.columns.tolist()}")
    logger.info(f"  -> Preview:\n{df}")
    
    return df


# ---- Funciones auxiliares ------------------------------------------------------------------


def _ensure_iterable(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_opening_hours(opening_hours: Any, default_range: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(opening_hours, str) and "-" in opening_hours:
        start, end = opening_hours.split("-", 1)
        return _time_str_to_int(start.strip()), _time_str_to_int(end.strip())
    if isinstance(opening_hours, dict):
        start = opening_hours.get("open") or opening_hours.get("from")
        end = opening_hours.get("close") or opening_hours.get("to")
        if start and end:
            return _time_str_to_int(str(start)), _time_str_to_int(str(end))
    return default_range


def _time_str_to_int(value: str) -> int:
    try:
        parts = value.split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        return hours * 100 + minutes
    except (ValueError, IndexError):
        return 800


def _estimate_total_cost(pois: List[POI]) -> float:
    if not pois:
        return 500.0
    return sum(p.entry_cost + p.stay_cost_per_hour * (p.duracion_visita_min / 60.0) for p in pois)


def _estimate_total_time(pois: List[POI]) -> int:
    if not pois:
        return 720
    return sum(p.duracion_visita_min for p in pois)


def _find_closest_poi_to_location(pois: List[POI], location: Dict[str, Any]) -> Optional[POI]:
    """Encuentra el POI más cercano a una ubicación específica."""
    if not pois:
        return None
    
    target_lat = _coerce_float(location.get("latitude", 0.0))
    target_lng = _coerce_float(location.get("longitude", 0.0))
    target_location = (target_lat, target_lng)
    
    closest_poi = pois[0]
    min_distance = _haversine(closest_poi.position, target_location)
    
    for poi in pois[1:]:
        distance = _haversine(poi.position, target_location)
        if distance < min_distance:
            min_distance = distance
            closest_poi = poi
    
    return closest_poi


def _haversine(origin: Tuple[float, float], target: Tuple[float, float]) -> float:
    """Calcula la distancia aproximada en kilómetros entre dos coordenadas."""

    lat1, lon1 = origin
    lat2, lon2 = target

    radius = 6371  # Radio terrestre en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c