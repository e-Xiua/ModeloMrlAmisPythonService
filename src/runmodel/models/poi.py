from dataclasses import dataclass
from typing import Tuple, Dict
import random
import numpy as np

@dataclass
class POI:
    """Punto de Interés con todas las propiedades necesarias para MRL-AMIS"""
    id: str
    name: str
    position: Tuple[float, float]  # (lat, lng)
    stay_cost_per_hour: float = 0.0
    entry_cost: float = 0.0
    co2_per_hour: float = 0.0  # Emisiones de CO2 por hora de estadía
    preferencia: float = 0.0  # Preferencia base del POI (0-100)
    sust_ambiental: float = 0.0  # Impacto ambiental del POI (0-100, mayor es mejor)
    sust_economica: float = 0.0  # Impacto económico del POI (0-100, mayor es mejor)
    sust_social: float = 0.0  # Impacto social del POI (0-100, mayor es mejor)
    riesgo_salud: float = 0.0  # Riesgo de salud asociado al POI (0-100, menor es mejor)
    riesgo_accidente: float = 0.0  # Riesgo de accidentes (0-100, menor es mejor)
    horario_apertura: float = 800  # Formato HHMM
    horario_cierre: float = 1800  # Formato HHMM
    duracion_visita_min: int = 60  # Duración recomendada de visita en minutos
    seguridad_poi: float = 80.0  # Nivel de seguridad del POI (0-100, mayor es mejor)
    capacidad_maxima: int = 100  # Capacidad máxima del POI
    tipo_poi: str = "general"  # cultura, aventura, naturaleza, gastronomia, etc.

    def to_dict(self) -> Dict:
        """Convierte POI a diccionario para compatibilidad con código existente"""
        return {
            'id': self.id,
            'name': self.name,
            'lat': self.position[0],
            'lng': self.position[1],
            'stay_cost_per_hour': self.stay_cost_per_hour,
            'entry_cost': self.entry_cost,
            'co2_per_hour': self.co2_per_hour,
            'preferencia': self.preferencia,
            'sust_ambiental': self.sust_ambiental,
            'sust_economica': self.sust_economica,
            'sust_social': self.sust_social,
            'riesgo_salud': self.riesgo_salud,
            'riesgo_accidente': self.riesgo_accidente,
            'horario_apertura': self.horario_apertura,
            'horario_cierre': self.horario_cierre,
            'duracion_visita_min': self.duracion_visita_min,
            'seguridad_poi': self.seguridad_poi,
            'capacidad_maxima': self.capacidad_maxima,
            'tipo_poi': self.tipo_poi
        }

    @property
    def combined_sustainability(self) -> float:
        """Calcula sostenibilidad combinada"""
        return (self.sust_ambiental + self.sust_economica + self.sust_social) / 3

    @property
    def total_risk(self) -> float:
        """Calcula riesgo total"""
        return (self.riesgo_salud + self.riesgo_accidente) / 2

    def get_preference_for_group(self, group_type: str) -> float:
        """
        Calcula preferencia específica para un tipo de grupo
        Esto simula las preferencias diferenciadas por grupo
        """
        base_preference = self.preferencia
        
        # Modificadores según tipo de POI y grupo
        modifiers = {
            'aventurero': {
                'aventura': 1.3,
                'naturaleza': 1.2,
                'cultura': 0.8,
                'gastronomia': 0.9
            },
            'cultural': {
                'cultura': 1.4,
                'gastronomia': 1.1,
                'aventura': 0.7,
                'naturaleza': 0.9
            },
            'ecoturista': {
                'naturaleza': 1.5,
                'aventura': 1.1,
                'cultura': 0.8,
                'gastronomia': 0.8
            },
            'familiar': {
                'cultura': 1.1,
                'naturaleza': 1.0,
                'aventura': 0.6,
                'gastronomia': 1.2
            },
            'lujo': {
                'gastronomia': 1.3,
                'cultura': 1.1,
                'aventura': 0.8,
                'naturaleza': 0.9
            },
            'backpacker': {
                'aventura': 1.2,
                'naturaleza': 1.1,
                'cultura': 1.0,
                'gastronomia': 0.7
            }
        }
        
        modifier = modifiers.get(group_type, {}).get(self.tipo_poi, 1.0)
        
        # Agregar algo de aleatoriedad para simular preferencias individuales
        random_factor = random.uniform(0.9, 1.1)
        
        return min(100, base_preference * modifier * random_factor)

def create_sample_pois() -> list[POI]:
    """Crea una lista de POIs basada en datos reales proporcionados"""
    
    # Datos reales de los POIs con costos de experiencias integrados
    real_pois_data = [
        {"poi_id": "1", "nombre": "Hotel Los Lagos Spa y Resort", "latitud": 10.4882, "longitud": -84.6866, 
         "preferencia": 80, "sust_ambiental": 70, "sust_social": 75, "sust_economica": 83, "riesgo_salud": 30,
         "horario_apertura": 480, "horario_cierre": 1020, "duracion_visita_min": 180, "seguridad_poi": 90, 
         "capacidad_max": 250, "categoria": "Hotel", "costo_nacional": 47, "costo_extranjero": 47},
        {"poi_id": "2", "nombre": "Tabacón Grand Spa Thermal Resort", "latitud": 10.4919, "longitud": -84.7216, 
         "preferencia": 90, "sust_ambiental": 84, "sust_social": 80, "sust_economica": 70, "riesgo_salud": 40,
         "horario_apertura": 600, "horario_cierre": 1320, "duracion_visita_min": 165, "seguridad_poi": 95, 
         "capacidad_max": 200, "categoria": "Termales", "costo_nacional": 50, "costo_extranjero": 99},
        {"poi_id": "3", "nombre": "Hotel Arenal Manoa y Hot Springs", "latitud": 10.5010, "longitud": -84.6800, 
         "preferencia": 90, "sust_ambiental": 75, "sust_social": 74, "sust_economica": 81, "riesgo_salud": 30,
         "horario_apertura": 540, "horario_cierre": 1080, "duracion_visita_min": 120, "seguridad_poi": 85, 
         "capacidad_max": 160, "categoria": "Hotel", "costo_nacional": 190, "costo_extranjero": 205},
        {"poi_id": "4", "nombre": "Hotel Arenal Nayara", "latitud": 10.5035, "longitud": -84.6868, 
         "preferencia": 98, "sust_ambiental": 80, "sust_social": 75, "sust_economica": 79, "riesgo_salud": 35,
         "horario_apertura": 540, "horario_cierre": 1080, "duracion_visita_min": 115, "seguridad_poi": 95, 
         "capacidad_max": 100, "categoria": "Hotel", "costo_nacional": 165, "costo_extranjero": 205},
        {"poi_id": "5", "nombre": "Volcano Lodge", "latitud": 10.4988, "longitud": -84.6857, 
         "preferencia": 90, "sust_ambiental": 89, "sust_social": 76, "sust_economica": 80, "riesgo_salud": 40,
         "horario_apertura": 480, "horario_cierre": 960, "duracion_visita_min": 90, "seguridad_poi": 85, 
         "capacidad_max": 130, "categoria": "Hotel", "costo_nacional": 20, "costo_extranjero": 40},
        {"poi_id": "6", "nombre": "Hotel Arenal Kioro Suites & Spa", "latitud": 10.4916, "longitud": -84.7114, 
         "preferencia": 90, "sust_ambiental": 65, "sust_social": 75, "sust_economica": 85, "riesgo_salud": 35,
         "horario_apertura": 480, "horario_cierre": 1020, "duracion_visita_min": 180, "seguridad_poi": 95, 
         "capacidad_max": 140, "categoria": "Hotel", "costo_nacional": 27, "costo_extranjero": 35},
        {"poi_id": "7", "nombre": "Hotel Montaña de Fuego", "latitud": 10.5045, "longitud": -84.7024, 
         "preferencia": 80, "sust_ambiental": 71, "sust_social": 79, "sust_economica": 75, "riesgo_salud": 50,
         "horario_apertura": 540, "horario_cierre": 1080, "duracion_visita_min": 135, "seguridad_poi": 80, 
         "capacidad_max": 110, "categoria": "Hotel", "costo_nacional": 180, "costo_extranjero": 180},
        {"poi_id": "8", "nombre": "Biosfera", "latitud": 10.4575, "longitud": -84.6624, 
         "preferencia": 98, "sust_ambiental": 95, "sust_social": 90, "sust_economica": 60, "riesgo_salud": 55,
         "horario_apertura": 420, "horario_cierre": 1020, "duracion_visita_min": 115, "seguridad_poi": 75, 
         "capacidad_max": 200, "categoria": "Reserva Natural", "costo_nacional": 10, "costo_extranjero": 25},
        {"poi_id": "9", "nombre": "Termales Los Laureles", "latitud": 10.4890, "longitud": -84.6771, 
         "preferencia": 90, "sust_ambiental": 85, "sust_social": 79, "sust_economica": 70, "riesgo_salud": 40,
         "horario_apertura": 480, "horario_cierre": 1200, "duracion_visita_min": 120, "seguridad_poi": 80, 
         "capacidad_max": 150, "categoria": "Termales", "costo_nacional": 11, "costo_extranjero": 12},
        {"poi_id": "10", "nombre": "Ecotermales La Fortuna", "latitud": 10.4841, "longitud": -84.6740, 
         "preferencia": 90, "sust_ambiental": 90, "sust_social": 81, "sust_economica": 75, "riesgo_salud": 45,
         "horario_apertura": 600, "horario_cierre": 1290, "duracion_visita_min": 115, "seguridad_poi": 85, 
         "capacidad_max": 180, "categoria": "Termales", "costo_nacional": 49, "costo_extranjero": 65},
        {"poi_id": "11", "nombre": "Baldi Hot Springs", "latitud": 10.4833, "longitud": -84.6800, 
         "preferencia": 70, "sust_ambiental": 79, "sust_social": 80, "sust_economica": 69, "riesgo_salud": 50,
         "horario_apertura": 540, "horario_cierre": 1320, "duracion_visita_min": 150, "seguridad_poi": 80, 
         "capacidad_max": 120, "categoria": "Termales", "costo_nacional": 55, "costo_extranjero": 72},
        {"poi_id": "12", "nombre": "Paradise Hot Springs", "latitud": 10.4847, "longitud": -84.6807, 
         "preferencia": 90, "sust_ambiental": 81, "sust_social": 78, "sust_economica": 70, "riesgo_salud": 40,
         "horario_apertura": 660, "horario_cierre": 1260, "duracion_visita_min": 120, "seguridad_poi": 85, 
         "capacidad_max": 200, "categoria": "Termales", "costo_nacional": 30, "costo_extranjero": 47},
        {"poi_id": "13", "nombre": "Termalitas del Arenal", "latitud": 10.4813, "longitud": -84.6765, 
         "preferencia": 80, "sust_ambiental": 91, "sust_social": 83, "sust_economica": 75, "riesgo_salud": 45,
         "horario_apertura": 540, "horario_cierre": 1260, "duracion_visita_min": 115, "seguridad_poi": 75, 
         "capacidad_max": 150, "categoria": "Termales", "costo_nacional": 9, "costo_extranjero": 10},
        {"poi_id": "14", "nombre": "The Spring Resort & Spa at Arenal", "latitud": 10.5193, "longitud": -84.6878, 
         "preferencia": 96, "sust_ambiental": 72, "sust_social": 75, "sust_economica": 80, "riesgo_salud": 40,
         "horario_apertura": 480, "horario_cierre": 1020, "duracion_visita_min": 120, "seguridad_poi": 95, 
         "capacidad_max": 210, "categoria": "Hotel", "costo_nacional": 85, "costo_extranjero": 110},
        {"poi_id": "15", "nombre": "Arenal Paraiso Resort Spa & Thermo Mineral Hot Springs", "latitud": 10.5025, "longitud": -84.6939, 
         "preferencia": 90, "sust_ambiental": 69, "sust_social": 73, "sust_economica": 82, "riesgo_salud": 45,
         "horario_apertura": 480, "horario_cierre": 1080, "duracion_visita_min": 150, "seguridad_poi": 85, 
         "capacidad_max": 220, "categoria": "Hotel", "costo_nacional": 79, "costo_extranjero": 99}
    ]
    
    sample_pois = []
    
    for poi_data in real_pois_data:
        # Determinar tipo de POI basado en categoría
        if poi_data["categoria"] == "Hotel":
            tipo_poi = "cultura"
        elif poi_data["categoria"] == "Termales":
            tipo_poi = "naturaleza"
        elif poi_data["categoria"] == "Reserva Natural":
            tipo_poi = "aventura"
        else:
            tipo_poi = "general"
        
        # Calcular costos por hora basados en duración de visita
        duracion_horas = poi_data["duracion_visita_min"] / 60
        stay_cost_per_hour = poi_data["costo_nacional"] / duracion_horas if duracion_horas > 0 else 0
        
        poi = POI(
            id=poi_data["poi_id"],
            name=poi_data["nombre"],
            position=(poi_data["latitud"], poi_data["longitud"]),
            stay_cost_per_hour=stay_cost_per_hour,
            entry_cost=0.0,  # Incluido en costo de experiencia
            co2_per_hour=0.3,  # Valor base, se puede ajustar
            preferencia=poi_data["preferencia"],
            sust_ambiental=poi_data["sust_ambiental"],
            sust_economica=poi_data["sust_economica"],
            sust_social=poi_data["sust_social"],
            riesgo_salud=poi_data["riesgo_salud"],
            riesgo_accidente=30,  # Valor base
            horario_apertura=poi_data["horario_apertura"],
            horario_cierre=poi_data["horario_cierre"],
            duracion_visita_min=poi_data["duracion_visita_min"],
            seguridad_poi=poi_data["seguridad_poi"],
            capacidad_maxima=poi_data["capacidad_max"],
            tipo_poi=tipo_poi
        )
        
        sample_pois.append(poi)
    
    return sample_pois
    
