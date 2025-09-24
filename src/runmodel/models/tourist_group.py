from dataclasses import dataclass
from typing import Dict, List
import random

@dataclass
class TouristGroup:
    """Grupo de turistas con todas las propiedades necesarias para MRL-AMIS"""
    grupo_id: str
    tipo_turista: str
    tamano_grupo: int  # Tamaño del grupo
    origen: str  # POI inicial (ID)
    destino: str  # POI final (ID) 
    presupuesto_max_usd: float  # Presupuesto total por persona en USD
    presupuesto_min_usd: float  # Presupuesto mínimo por persona
    sostenibilidad_min: float  # Nivel mínimo de sostenibilidad requerido (0-100)
    limite_max_dist_km: float  # Distancia máxima total en km
    tiempo_disponible_min: int  # Tiempo total disponible en minutos
    preferencias_tipos: Dict[str, float]  # Preferencias por tipos de POI
    nivel_aventura: float = 50.0  # Nivel de aventura deseado (0-100)
    sensibilidad_costo: float = 50.0  # Sensibilidad al costo (0-100, mayor = más sensible)
    conciencia_ambiental: float = 50.0  # Conciencia ambiental (0-100)
    tolerancia_riesgo: float = 50.0  # Tolerancia al riesgo (0-100)

    def to_dict(self) -> Dict:
        """Convierte TouristGroup a diccionario para compatibilidad"""
        return {
            'grupo_id': self.grupo_id,
            'tipo_turista': self.tipo_turista,
            'tamano_grupo': self.tamano_grupo,
            'origen': self.origen,
            'destino': self.destino,
            'presupuesto_max_usd': self.presupuesto_max_usd,
            'presupuesto_min_usd': self.presupuesto_min_usd,
            'sostenibilidad_min': self.sostenibilidad_min,
            'limite_max_dist_km': self.limite_max_dist_km,
            'tiempo_max_min': self.tiempo_disponible_min,  # Renamed for compatibility
            'preferencias_tipos': self.preferencias_tipos,
            'nivel_aventura': self.nivel_aventura,
            'sensibilidad_costo': self.sensibilidad_costo,
            'conciencia_ambiental': self.conciencia_ambiental,
            'tolerancia_riesgo': self.tolerancia_riesgo
        }

    def calculate_poi_preference(self, poi_type: str, base_preference: float) -> float:
        """
        Calcula la preferencia específica de este grupo por un POI
        """
        # Preferencia base del grupo por el tipo de POI
        type_preference = self.preferencias_tipos.get(poi_type, 50.0)
        
        # Combinar preferencia del grupo por tipo con preferencia base del POI
        combined_preference = (type_preference * 0.6) + (base_preference * 0.4)
        
        # Aplicar modificadores según características del grupo
        if poi_type == 'aventura':
            combined_preference *= (self.nivel_aventura / 100) * 1.2 + 0.8
        
        if poi_type == 'naturaleza':
            combined_preference *= (self.conciencia_ambiental / 100) * 1.3 + 0.7
        
        # Normalizar a rango 0-100
        return max(0, min(100, combined_preference))

def create_sample_tourist_groups() -> List[TouristGroup]:
    """
    Crea grupos de turistas basados en datos reales proporcionados
    """
    import random
    
    # Datos reales de los grupos
    real_groups_data = [
        {"grupo_id": "grupo_1", "tipo_turista": "nacional", "tamano_nk": 10, "origen": "1", "destino": "7", 
         "tiempo_max_min": 720, "presupuesto_max_usd": 700, "sostenibilidad_min": 50, "limite_max_dist_km": 50},
        {"grupo_id": "grupo_2", "tipo_turista": "extranjero", "tamano_nk": 20, "origen": "1", "destino": "10", 
         "tiempo_max_min": 1440, "presupuesto_max_usd": 900, "sostenibilidad_min": 70, "limite_max_dist_km": 75},
        {"grupo_id": "grupo_3", "tipo_turista": "extranjero", "tamano_nk": 15, "origen": "1", "destino": "15", 
         "tiempo_max_min": 720, "presupuesto_max_usd": 750, "sostenibilidad_min": 90, "limite_max_dist_km": 100},
        {"grupo_id": "grupo_4", "tipo_turista": "nacional", "tamano_nk": 17, "origen": "1", "destino": "8", 
         "tiempo_max_min": 600, "presupuesto_max_usd": 550, "sostenibilidad_min": 80, "limite_max_dist_km": 100},
        {"grupo_id": "grupo_5", "tipo_turista": "extranjero", "tamano_nk": 21, "origen": "1", "destino": "6", 
         "tiempo_max_min": 720, "presupuesto_max_usd": 650, "sostenibilidad_min": 60, "limite_max_dist_km": 80},
        {"grupo_id": "grupo_6", "tipo_turista": "nacional", "tamano_nk": 13, "origen": "1", "destino": "12", 
         "tiempo_max_min": 840, "presupuesto_max_usd": 675, "sostenibilidad_min": 50, "limite_max_dist_km": 110}
    ]
    
    tourist_groups = []
    
    for group_data in real_groups_data:
        # Generar preferencias por tipo de POI basadas en el tipo de turista
        if group_data["tipo_turista"] == "nacional":
            preferencias = {
                "cultura": random.uniform(60, 80),
                "naturaleza": random.uniform(70, 90),
                "aventura": random.uniform(40, 70),
                "gastronomia": random.uniform(60, 85),
                "general": random.uniform(50, 70)
            }
        else:  # extranjero
            preferencias = {
                "cultura": random.uniform(70, 90),
                "naturaleza": random.uniform(80, 95),
                "aventura": random.uniform(60, 85),
                "gastronomia": random.uniform(50, 75),
                "general": random.uniform(60, 80)
            }
        
        # Calcular presupuesto mínimo como porcentaje del máximo
        presupuesto_min = group_data["presupuesto_max_usd"] * 0.6
        
        # Crear el grupo turístico
        group = TouristGroup(
            grupo_id=group_data["grupo_id"],
            tipo_turista=group_data["tipo_turista"],
            tamano_grupo=group_data["tamano_nk"],
            origen=group_data["origen"],
            destino=group_data["destino"],
            presupuesto_max_usd=group_data["presupuesto_max_usd"],
            presupuesto_min_usd=presupuesto_min,
            sostenibilidad_min=group_data["sostenibilidad_min"],
            limite_max_dist_km=group_data["limite_max_dist_km"],
            tiempo_disponible_min=group_data["tiempo_max_min"],
            preferencias_tipos=preferencias,
            nivel_aventura=random.uniform(30, 80),
            sensibilidad_costo=random.uniform(40, 80),
            conciencia_ambiental=group_data["sostenibilidad_min"] + random.uniform(-10, 10),
            tolerancia_riesgo=random.uniform(30, 70)
        )
        
        tourist_groups.append(group)
    
    return tourist_groups

def generate_group_poi_preferences(groups: List[TouristGroup], pois) -> Dict[str, Dict[str, float]]:
    """
    Genera matriz de preferencias de grupos por POIs específicos
    Retorna dict[grupo_id][poi_id] = preferencia
    """
    preferences = {}
    
    for group in groups:
        group_prefs = {}
        for poi in pois:
            # Calcular preferencia base del grupo por este POI
            base_pref = group.calculate_poi_preference(poi.tipo_poi, poi.preferencia)
            
            # Agregar variabilidad realista
            variation = random.uniform(0.85, 1.15)
            final_pref = base_pref * variation
            
            # Asegurar rango válido
            final_pref = max(0, min(100, final_pref))
            
            group_prefs[poi.id] = final_pref
            
        preferences[group.grupo_id] = group_prefs
    
    return preferences
