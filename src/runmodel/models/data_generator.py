import numpy as np
import pandas as pd
import random
import math
from typing import List, Dict, Tuple
from .poi import POI, create_sample_pois
from .tourist_group import TouristGroup, create_sample_tourist_groups, generate_group_poi_preferences

class DataGenerator:
    """Generador de datos sintéticos para reemplazar la carga desde Excel"""
    
    def __init__(self, num_pois: int = 15, seed: int = 42):
        """
        Inicializa el generador de datos
        
        Args:
            num_pois: Número de POIs a generar
            seed: Semilla para reproducibilidad
        """
        self.num_pois = num_pois
        random.seed(seed)
        np.random.seed(seed)
        
        # Generar POIs y grupos
        self.pois = self._generate_pois()
        self.tourist_groups = create_sample_tourist_groups()
        
    def _generate_pois(self) -> List[POI]:
        """Genera POIs usando los de ejemplo o creando adicionales si se necesitan más"""
        base_pois = create_sample_pois()
        
        if len(base_pois) >= self.num_pois:
            return base_pois[:self.num_pois]
        
        # Si necesitamos más POIs, generar adicionales
        additional_pois = []
        for i in range(len(base_pois), self.num_pois):
            poi = self._generate_random_poi(str(i + 1))
            additional_pois.append(poi)
            
        return base_pois + additional_pois
    
    def _generate_random_poi(self, poi_id: str) -> POI:
        """Genera un POI aleatorio con valores realistas"""
        # Coordenadas dentro de Costa Rica aproximadamente
        lat = random.uniform(8.0, 11.2)
        lng = random.uniform(-85.9, -82.5)
        
        # Tipos de POI disponibles
        tipos = ['cultura', 'aventura', 'naturaleza', 'gastronomia']
        tipo = random.choice(tipos)
        
        # Nombres genéricos
        nombres_base = {
            'cultura': ['Museo', 'Centro Histórico', 'Iglesia', 'Plaza', 'Teatro'],
            'aventura': ['Canopy', 'Rafting', 'Rappel', 'Tirolina', 'Escalada'],
            'naturaleza': ['Parque', 'Reserva', 'Volcán', 'Bosque', 'Río'],
            'gastronomia': ['Restaurante', 'Mercado', 'Café', 'Finca', 'Sodas']
        }
        
        base_name = random.choice(nombres_base[tipo])
        name = f"{base_name} {poi_id}"
        
        return POI(
            id=poi_id,
            name=name,
            position=(lat, lng),
            stay_cost_per_hour=random.uniform(2.0, 15.0),
            entry_cost=random.uniform(0.0, 30.0),
            co2_per_hour=random.uniform(0.1, 0.8),
            preferencia=random.uniform(60, 95),
            sust_ambiental=random.uniform(50, 100),
            sust_economica=random.uniform(50, 100),
            sust_social=random.uniform(50, 100),
            riesgo_salud=random.uniform(5, 35),
            riesgo_accidente=random.uniform(10, 40),
            horario_apertura=random.choice([600, 700, 800]),
            horario_cierre=random.choice([1700, 1800, 1900, 2000, 2200]),
            duracion_visita_min=random.randint(60, 300),
            seguridad_poi=random.uniform(60, 95),
            capacidad_maxima=random.randint(50, 500),
            tipo_poi=tipo
        )
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """Calcula matriz de distancias usando fórmula de Haversine"""
        n = len(self.pois)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self._haversine_distance(
                        self.pois[i].position,
                        self.pois[j].position
                    )
                    distance_matrix[i, j] = dist
                    
        return distance_matrix
    
    def calculate_travel_time_matrix(self, avg_speed_kmh: float = 50.0) -> np.ndarray:
        """Calcula matriz de tiempos de viaje basada en distancias"""
        distance_matrix = self.calculate_distance_matrix()
        
        # Convertir distancia a tiempo (en minutos)
        # Agregar factor de corrección para rutas no directas y paradas
        correction_factor = 1.3
        time_matrix = (distance_matrix / avg_speed_kmh) * 60 * correction_factor
        
        return time_matrix
    
    def calculate_cost_matrix(self, cost_per_km: float = 0.15) -> np.ndarray:
        """Calcula matriz de costos de transporte"""
        distance_matrix = self.calculate_distance_matrix()
        
        # Costo base de transporte más costo por distancia
        base_cost = 2.0  # Costo fijo por viaje
        cost_matrix = base_cost + (distance_matrix * cost_per_km)
        
        # Agregar costos de entrada de cada POI de destino
        for i in range(len(self.pois)):
            for j in range(len(self.pois)):
                if i != j:
                    cost_matrix[i, j] += self.pois[j].entry_cost
                    
        return cost_matrix
    
    def calculate_co2_matrix(self, co2_per_km: float = 0.12) -> np.ndarray:
        """Calcula matriz de emisiones de CO2"""
        distance_matrix = self.calculate_distance_matrix()
        
        # CO2 por transporte más CO2 por estadía en destino
        co2_matrix = distance_matrix * co2_per_km
        
        # Agregar CO2 de estadía (1 hora promedio)
        for i in range(len(self.pois)):
            for j in range(len(self.pois)):
                if i != j:
                    co2_matrix[i, j] += self.pois[j].co2_per_hour
                    
        return co2_matrix
    
    def calculate_accident_risk_matrix(self) -> np.ndarray:
        """Calcula matriz de riesgo de accidentes adaptada al número de POIs"""
        n = len(self.pois)
        
        # Matriz de riesgo de accidentes real proporcionada (para referencia con 15 POIs)
        real_accident_risk_reference = [
            [0, 40, 20, 20, 20, 40, 20, 70, 20, 20, 20, 20, 20, 70, 20],
            [40, 0, 40, 40, 40, 20, 20, 70, 40, 40, 40, 40, 40, 70, 40],
            [20, 40, 0, 20, 20, 40, 20, 70, 20, 20, 20, 20, 20, 70, 20],
            [20, 40, 20, 0, 20, 40, 20, 70, 20, 20, 20, 20, 20, 70, 20],
            [20, 40, 20, 20, 0, 40, 20, 70, 20, 20, 20, 20, 20, 70, 20],
            [40, 20, 40, 40, 40, 0, 20, 70, 40, 40, 40, 40, 40, 70, 40],
            [20, 20, 20, 20, 20, 20, 0, 70, 20, 20, 20, 20, 20, 70, 20],
            [70, 70, 70, 70, 70, 70, 70, 0, 70, 70, 70, 70, 70, 70, 70],
            [20, 40, 20, 20, 20, 40, 20, 70, 0, 20, 20, 20, 20, 70, 40],
            [20, 40, 20, 20, 20, 40, 20, 70, 20, 0, 20, 20, 20, 70, 40],
            [20, 40, 20, 20, 20, 40, 20, 70, 20, 20, 0, 20, 20, 70, 40],
            [20, 40, 20, 20, 20, 40, 20, 70, 20, 20, 20, 0, 20, 70, 40],
            [20, 40, 20, 20, 20, 40, 20, 70, 20, 20, 20, 20, 0, 70, 40],
            [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 0, 70],
            [20, 40, 20, 20, 20, 40, 20, 70, 40, 40, 40, 40, 40, 70, 0]
        ]
        
        # Si tenemos el número exacto de POIs de referencia, usar la matriz real
        if n == 15:
            return np.array(real_accident_risk_reference, dtype=float)
        
        # Para otros números de POIs, generar matriz adaptiva
        accident_risk_matrix = np.zeros((n, n))
        
        # Usar los primeros n POIs de la matriz de referencia si están disponibles
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    accident_risk_matrix[i, j] = real_accident_risk_reference[i][j]
        else:
            # Si necesitamos más POIs que los de referencia, extender con valores simulados
            for i in range(n):
                for j in range(n):
                    if i == j:
                        accident_risk_matrix[i, j] = 0  # Sin riesgo consigo mismo
                    elif i < 15 and j < 15:
                        # Usar datos reales para los primeros 15 POIs
                        accident_risk_matrix[i, j] = real_accident_risk_reference[i][j]
                    else:
                        # Generar valores simulados para POIs adicionales
                        # Basado en patrones de la matriz real: riesgo bajo (20), medio (40), alto (70)
                        risk_levels = [20, 40, 70]
                        # Usar una distribución sesgada hacia riesgo bajo y medio
                        weights = [0.6, 0.3, 0.1]  # 60% bajo, 30% medio, 10% alto
                        accident_risk_matrix[i, j] = np.random.choice(risk_levels, p=weights)
        
        return accident_risk_matrix
    
    def _haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calcula distancia en km usando fórmula de Haversine"""
        R = 6371  # Radio de la Tierra en km
        
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def generate_all_data(self) -> Dict:
        """
        Genera todos los datos necesarios para MRL-AMIS
        Retorna estructura compatible con cargar_datos()
        """
        print("Generando datos sintéticos para MRL-AMIS...")
        
        # Crear DataFrames de POIs usando índices secuenciales
        pois_data = []
        for i, poi in enumerate(self.pois):
            poi_dict = poi.to_dict()
            poi_dict['id'] = str(i+1)  # Asegurar índices secuenciales consistentes
            pois_data.append(poi_dict)
        pois_df = pd.DataFrame(pois_data).set_index('id')
        
        # Crear DataFrame de grupos turísticos
        groups_data = []
        for group in self.tourist_groups:
            groups_data.append(group.to_dict())
        groups_df = pd.DataFrame(groups_data).set_index('grupo_id')
        
        # Generar matrices usando índices secuenciales consistentes
        poi_ids = [str(i+1) for i in range(len(self.pois))]
        
        distances_df = pd.DataFrame(
            self.calculate_distance_matrix(),
            index=poi_ids,
            columns=poi_ids
        )
        
        travel_times_df = pd.DataFrame(
            self.calculate_travel_time_matrix(),
            index=poi_ids,
            columns=poi_ids
        )
        
        costs_df = pd.DataFrame(
            self.calculate_cost_matrix(),
            index=poi_ids,
            columns=poi_ids
        )
        
        co2_df = pd.DataFrame(
            self.calculate_co2_matrix(),
            index=poi_ids,
            columns=poi_ids
        )
        
        accident_risk_df = pd.DataFrame(
            self.calculate_accident_risk_matrix(),
            index=poi_ids,
            columns=poi_ids
        )
        
        # Generar preferencias de grupos por POIs
        group_preferences = generate_group_poi_preferences(self.tourist_groups, self.pois)
        preferences_df = pd.DataFrame(group_preferences).T
        preferences_df.columns = poi_ids
        
        # Generar costos de experiencias basados en datos reales
        real_experience_costs = {
            "1": {"costo_nacional_exp_USD": 47, "costo_extranjero_exp_USD": 47},
            "2": {"costo_nacional_exp_USD": 50, "costo_extranjero_exp_USD": 99},
            "3": {"costo_nacional_exp_USD": 190, "costo_extranjero_exp_USD": 205},
            "4": {"costo_nacional_exp_USD": 165, "costo_extranjero_exp_USD": 205},
            "5": {"costo_nacional_exp_USD": 20, "costo_extranjero_exp_USD": 40},
            "6": {"costo_nacional_exp_USD": 27, "costo_extranjero_exp_USD": 35},
            "7": {"costo_nacional_exp_USD": 180, "costo_extranjero_exp_USD": 180},
            "8": {"costo_nacional_exp_USD": 10, "costo_extranjero_exp_USD": 25},
            "9": {"costo_nacional_exp_USD": 11, "costo_extranjero_exp_USD": 12},
            "10": {"costo_nacional_exp_USD": 49, "costo_extranjero_exp_USD": 65},
            "11": {"costo_nacional_exp_USD": 55, "costo_extranjero_exp_USD": 72},
            "12": {"costo_nacional_exp_USD": 30, "costo_extranjero_exp_USD": 47},
            "13": {"costo_nacional_exp_USD": 9, "costo_extranjero_exp_USD": 10},
            "14": {"costo_nacional_exp_USD": 85, "costo_extranjero_exp_USD": 110},
            "15": {"costo_nacional_exp_USD": 79, "costo_extranjero_exp_USD": 99}
        }
        
        experience_costs = {}
        for poi in self.pois:
            if poi.id in real_experience_costs:
                costs = real_experience_costs[poi.id]
                experience_costs[poi.id] = {
                    'costo_nacional_exp_USD': costs["costo_nacional_exp_USD"],
                    'costo_extranjero_exp_USD': costs["costo_extranjero_exp_USD"],
                    'costo_base': (costs["costo_nacional_exp_USD"] + costs["costo_extranjero_exp_USD"]) / 2,
                    'costo_premium': costs["costo_extranjero_exp_USD"] * 1.2,
                    'costo_economico': costs["costo_nacional_exp_USD"] * 0.8
                }
            else:
                # Fallback para POIs no encontrados
                base_cost = poi.entry_cost + (poi.stay_cost_per_hour * 2)
                experience_costs[poi.id] = {
                    'costo_nacional_exp_USD': base_cost * 0.7,
                    'costo_extranjero_exp_USD': base_cost * 1.2,
                    'costo_base': base_cost,
                    'costo_premium': base_cost * 1.5,
                    'costo_economico': base_cost * 0.7
                }
        
        costos_experiencia_df = pd.DataFrame(experience_costs).T
        
        # Parámetros generales basados en datos reales
        parametros_generales = {
            'consumo_combustible': 3.5,
            'costo_emisiones_co2': 0.0013,
            'emisiones_co2_km': 0.76,
            'peso_sust_ambiental': 0.6,
            'peso_sust_social': 0.3,
            'peso_sust_economica': 0.1,
            'inflacion': 0.0084,
            'tasa_cambio': 510,
            'peso_inflacion': 0.5,
            'peso_tasa_cambio': 0.5,
            'num_pois': len(self.pois),
            'num_grupos': len(self.tourist_groups),
            'velocidad_promedio_kmh': 50.0,
            'costo_combustible_por_km': 0.15,
            'factor_correccion_ruta': 1.3,
            'co2_por_km': 0.12,
            'tiempo_parada_min': 15,
            'distancia_maxima_diaria': 500.0
        }
        
        print(f"✓ Generados {len(self.pois)} POIs")
        print(f"✓ Generados {len(self.tourist_groups)} grupos turísticos")
        print(f"✓ Calculadas matrices {len(poi_ids)}x{len(poi_ids)}")
        print("Datos sintéticos generados exitosamente.")
        
        # Retornar estructura compatible con cargar_datos()
        return {
            'pois': pois_df,
            'tourist_groups': groups_df,
            'distances': distances_df,
            'travel_times': travel_times_df,
            'costs': costs_df,
            'co2_emission_cost': co2_df,
            'accident_risk': accident_risk_df,
            'preferencias_grupos_turistas': preferences_df,
            'costos_experiencia': costos_experiencia_df,
            'parametros_generales': parametros_generales
        }

def generate_synthetic_data(num_pois: int = 15, seed: int = 42) -> Dict:
    """
    Función de conveniencia para generar datos sintéticos
    
    Args:
        num_pois: Número de POIs a generar
        seed: Semilla para reproducibilidad
        
    Returns:
        Dict con todos los datos necesarios para MRL-AMIS
    """
    generator = DataGenerator(num_pois=num_pois, seed=seed)
    return generator.generate_all_data()
