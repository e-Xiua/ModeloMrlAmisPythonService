import grpc
from concurrent import futures
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import queue
import threading
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

import sys
import os

 # Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # -> .../ModeloMrlAmisPythonService/src
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
from generated import route_optimization_pb2 
from generated import route_optimization_pb2_grpc 


# Importar funciones del modelo MRL-AMIS
from runmodel.models.data_generator import generate_synthetic_data
from AgenteQLearning.QLearningAgent import QLearningAgent
from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import find_pareto_front, update_population
from generacionWorkPackages.workPackages import decodificar_wp, generar_work_packages
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
from estado_y_recompensa_rl.definir_comportamiento import calculate_reward, get_state
from intelligence_boxes.definir_intelligence_boxes import (
    ib_random_perturbation,
    ib_swap_mutation,
    ib_inversion_mutation,
    ib_guided_perturbation,
    ib_local_search,
    ib_diversity_mutation
)
from analisis_multi_objetivo_y_metricos.hypervolume import calculate_hypervolume
from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import calculate_average_ratio_pareto, spacing_metric

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enums y dataclasses para gestión de trabajos
class JobStatus(Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class OptimizationJob:
    job_id: str
    route_id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime = None
    completed_at: datetime = None
    progress: float = 0.0
    result: Dict[str, Any] = None
    error_message: str = None
    process_id: int = None
    estimated_completion: datetime = None

class JobQueue:
    """Sistema de colas para gestionar trabajos de optimización"""
    
    def __init__(self, max_concurrent_jobs=2, max_queue_size=50):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        self.job_queue = queue.Queue(maxsize=max_queue_size)
        self.active_jobs = {}  # job_id -> OptimizationJob
        self.completed_jobs = {}  # job_id -> OptimizationJob
        self.lock = threading.RLock()
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
        # Inicializar workers
        for i in range(max_concurrent_jobs):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.worker_threads.append(worker)
            
        logger.info(f"JobQueue inicializada con {max_concurrent_jobs} workers concurrentes")
    
    def submit_job(self, job_data: Dict[str, Any], request) -> str:
        """Enviar trabajo a la cola"""
        job_id = str(uuid.uuid4())
        
        job = OptimizationJob(
            job_id=job_id,
            route_id=job_data.get('route_id', f'route_{job_id[:8]}'),
            user_id=job_data.get('user_id', 'unknown'),
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=5)
        )
        
        try:
            # Crear trabajo serializable
            work_item = {
                'job': job,
                'job_data': job_data,
                'request_data': self._serialize_grpc_request(request)
            }
            
            self.job_queue.put(work_item, timeout=1.0)
            
            with self.lock:
                self.active_jobs[job_id] = job
            
            logger.info(f"Trabajo {job_id} añadido a la cola. Posición en cola: {self.job_queue.qsize()}")
            return job_id
            
        except queue.Full:
            logger.error(f"Cola llena. No se puede procesar trabajo {job_id}")
            raise Exception("Sistema ocupado. Intente más tarde.")
    
    def get_job_status(self, job_id: str) -> OptimizationJob:
        """Obtener estado del trabajo"""
        with self.lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            else:
                return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar trabajo"""
        with self.lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
                    logger.info(f"Trabajo {job_id} cancelado")
                    return True
        return False
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Obtener información de la cola"""
        with self.lock:
            return {
                'queue_size': self.job_queue.qsize(),
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len(self.completed_jobs),
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'workers_status': [not t.is_alive() for t in self.worker_threads]
            }
    
    def _worker_loop(self, worker_id: int):
        """Loop principal del worker"""
        logger.info(f"Worker {worker_id} iniciado")
        
        while not self.shutdown_event.is_set():
            try:
                # Obtener trabajo de la cola
                work_item = self.job_queue.get(timeout=1.0)
                job = work_item['job']
                job_data = work_item['job_data']
                request_data = work_item['request_data']
                
                logger.info(f"Worker {worker_id} procesando trabajo {job.job_id}")
                
                # Actualizar estado
                with self.lock:
                    job.status = JobStatus.PROCESSING
                    job.started_at = datetime.now()
                
                # Procesar trabajo
                try:
                    result = self._execute_mrl_amis_in_process(job_data, request_data, job.job_id)
                    
                    with self.lock:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        job.progress = 100.0
                        
                        # Mover a completados
                        self.completed_jobs[job.job_id] = job
                        if job.job_id in self.active_jobs:
                            del self.active_jobs[job.job_id]
                    
                    logger.info(f"Trabajo {job.job_id} completado exitosamente")
                    
                except Exception as e:
                    logger.error(f"Error procesando trabajo {job.job_id}: {str(e)}")
                    
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now()
                        job.error_message = str(e)
                        
                        # Mover a completados
                        self.completed_jobs[job.job_id] = job
                        if job.job_id in self.active_jobs:
                            del self.active_jobs[job.job_id]
                
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en worker {worker_id}: {str(e)}")
        
        logger.info(f"Worker {worker_id} terminado")
    
    def _serialize_grpc_request(self, request) -> Dict[str, Any]:
        """Serializar solicitud gRPC para procesamiento"""
        return {
            'route_id': request.route_id,
            'user_id': request.user_id,
            'pois': [{
                'id': poi.id,
                'name': poi.name,
                'latitude': poi.latitude,
                'longitude': poi.longitude,
                'category': poi.category,
                'subcategory': poi.subcategory,
                'visit_duration': poi.visit_duration,
                'cost': poi.cost,
                'rating': poi.rating,
                'provider_id': poi.provider_id,
                'provider_name': poi.provider_name
            } for poi in request.pois],
            'preferences': {
                'optimize_for': request.preferences.optimize_for,
                'max_total_time': request.preferences.max_total_time,
                'max_total_cost': request.preferences.max_total_cost,
                'accessibility_required': request.preferences.accessibility_required
            } if request.preferences else {},
            'constraints': {
                'start_time': request.constraints.start_time,
                'lunch_break_required': request.constraints.lunch_break_required,
                'lunch_break_duration': request.constraints.lunch_break_duration
            } if request.constraints else {}
        }
    
    def _execute_mrl_amis_in_process(self, job_data: Dict[str, Any], request_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Ejecutar MRL-AMIS en proceso separado"""
        # Este método será implementado para usar ProcessPoolExecutor
        # Por ahora, simulamos el procesamiento
        import time
        start_time = time.time()
        
        # Simular progreso
        for i in range(10):
            time.sleep(0.5)
            progress = (i + 1) * 10
            with self.lock:
                if job_id in self.active_jobs:
                    self.active_jobs[job_id].progress = progress
        
        # Aquí iría la lógica real del modelo MRL-AMIS
        # Por simplicidad, retornamos un resultado mock
        execution_time = time.time() - start_time
        
        return {
            'optimized_route_id': f'optimized_{job_id[:8]}',
            'optimized_sequence': [],
            'total_distance_km': 45.7,
            'total_time_minutes': 280,
            'optimization_algorithm': 'MRL-AMIS',
            'optimization_score': 0.85,
            'execution_time_seconds': execution_time,
            'generated_at': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Cerrar sistema de colas"""
        logger.info("Cerrando sistema de colas...")
        self.shutdown_event.set()
        
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        logger.info("Sistema de colas cerrado")

class RouteOptimizationServicer(route_optimization_pb2_grpc.RouteOptimizationServiceServicer):
    
    def __init__(self):
        """Inicializar el servicio de optimización de rutas"""
        logger.info("Inicializando RouteOptimizationService...")
        
        # Configuración predeterminada del modelo MRL-AMIS
        self.default_config = {
            'num_work_packages': 100,
            'max_iterations': 100,
            'num_pois': 15,
            'max_pois_per_route': 10,
            'min_pois_per_route': 3,
            'num_routes_per_wp': 3,
            'max_duration_per_route': 720,
            'maximize_objectives_list': [True, False, False, True, False],  # [Preference+, Cost-, CO2-, Sustainability+, Risk-]
            'ref_point_hypervolume': [0, 1000, 1000, -1000, 1000]
        }
        
        # Configuración del agente RL
        self.rl_config = {
            'state_space_size': 7,
            'action_space_size': 6,
            'learning_rate': 0.2,
            'discount_factor': 0.85,
            'epsilon_start': 1.0,
            'epsilon_decay_rate': 0.995,
            'min_epsilon': 0.15
        }
        
        # Intelligence Boxes disponibles
        self.intelligence_boxes = {
            0: ib_random_perturbation,
            1: ib_swap_mutation,
            2: ib_inversion_mutation,
            3: ib_guided_perturbation,
            4: ib_local_search,
            5: ib_diversity_mutation
        }
        
        logger.info("RouteOptimizationService inicializado correctamente")
    
    def OptimizeRoute(self, request, context):
        """Método principal para optimizar rutas usando MRL-AMIS"""
        try:
            logger.info(f"=== SOLICITUD DE OPTIMIZACIÓN RECIBIDA ===")
            logger.info(f"Route ID: {request.route_id}")
            logger.info(f"User ID: {request.user_id}")
            logger.info(f"Número de POIs: {len(request.pois)}")
            
            start_time = time.time()
            
            # Convertir la solicitud gRPC a datos compatibles con el modelo
            data = self._convert_grpc_request_to_model_data(request)
            
            # Ejecutar el modelo MRL-AMIS
            results = self._execute_mrl_amis_model(data, request)
            
            execution_time = time.time() - start_time
            logger.info(f"Optimización completada en {execution_time:.2f} segundos")
            
            # Convertir resultados a respuesta gRPC
            response = self._convert_results_to_grpc_response(
                request.route_id, results, execution_time
            )
            
            logger.info(f"=== RESPUESTA DE OPTIMIZACIÓN ENVIADA ===")
            return response
            
        except Exception as e:
            logger.error(f"Error en optimización de ruta: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error interno del servidor: {str(e)}")
            
            return route_optimization_pb2.RouteOptimizationResponse(
                route_id=request.route_id,
                success=False,
                message=f"Error en optimización: {str(e)}"
            )
    
    def HealthCheck(self, request, context):
        """Verificar el estado del servicio"""
        return route_optimization_pb2.HealthResponse(
            is_healthy=True,
            status="Service is running",
            version="1.0.0"
        )
    
    def _convert_grpc_request_to_model_data(self, request) -> Dict[str, Any]:
        """Convertir solicitud gRPC a formato de datos del modelo MRL-AMIS"""
        logger.info("Convirtiendo solicitud gRPC a datos del modelo...")
        
        # Generar datos base sintéticos y luego reemplazar POIs con datos reales
        num_pois = len(request.pois)
        data = generate_synthetic_data(num_pois=num_pois, seed=42)
        
        # Actualizar POIs sintéticos con datos reales manteniendo estructura completa
        pois_sinteticos = data['pois'].copy()
        
        for i, poi in enumerate(request.pois):
            poi_id = str(i+1)  # Usar índices secuenciales para consistencia con matrices
            
            # Actualizar POI sintético con datos reales de la solicitud
            if poi_id in pois_sinteticos.index:
                # Mantener datos sintéticos y actualizar con datos reales disponibles
                pois_sinteticos.loc[poi_id, 'nombre'] = poi.name
                pois_sinteticos.loc[poi_id, 'latitud'] = poi.latitude  
                pois_sinteticos.loc[poi_id, 'longitud'] = poi.longitude
                pois_sinteticos.loc[poi_id, 'categoria'] = poi.category
                pois_sinteticos.loc[poi_id, 'subcategoria'] = poi.subcategory
                pois_sinteticos.loc[poi_id, 'duracion_visita_min'] = poi.visit_duration
                pois_sinteticos.loc[poi_id, 'costo'] = poi.cost
                pois_sinteticos.loc[poi_id, 'rating'] = poi.rating
                pois_sinteticos.loc[poi_id, 'descripcion'] = poi.description
                pois_sinteticos.loc[poi_id, 'accesibilidad'] = poi.accessibility
                pois_sinteticos.loc[poi_id, 'provider_id'] = poi.provider_id
                pois_sinteticos.loc[poi_id, 'provider_name'] = poi.provider_name
                pois_sinteticos.loc[poi_id, 'original_id'] = poi.id
                
                # Actualizar coordenadas para recálculo de matrices
                pois_sinteticos.loc[poi_id, 'lat'] = poi.latitude
                pois_sinteticos.loc[poi_id, 'lng'] = poi.longitude
        
        # Actualizar DataFrame de POIs en data manteniendo todas las columnas necesarias
        data['pois'] = pois_sinteticos
        
        # Actualizar matrices de distancias y tiempos basadas en coordenadas reales
        data = self._update_distance_and_time_matrices(data)
        
        # Crear información del grupo turístico basada en preferencias y restricciones
        grupo_info = self._create_grupo_info_from_request(request)
        data['grupo_info'] = grupo_info
        
        logger.info(f"Datos del modelo preparados: {len(data['pois'])} POIs")
        return data
    
    def _update_distance_and_time_matrices(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar matrices de distancias y tiempos basadas en coordenadas reales"""
        pois_df = data['pois']
        num_pois = len(pois_df)
        
        # Calcular matriz de distancias usando distancia euclidiana (simplificado)
        # En producción, se podría usar APIs como Google Maps o OpenStreetMap
        distances = np.zeros((num_pois, num_pois))
        travel_times = np.zeros((num_pois, num_pois))
        
        for i in range(num_pois):
            for j in range(num_pois):
                if i != j:
                    lat1, lon1 = pois_df.iloc[i]['latitud'], pois_df.iloc[i]['longitud']
                    lat2, lon2 = pois_df.iloc[j]['latitud'], pois_df.iloc[j]['longitud']
                    
                    # Calcular distancia euclidiana (aproximada)
                    # Convertir a kilómetros (aproximadamente 111 km por grado)
                    distance_km = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111
                    distances[i][j] = distance_km
                    
                    # Estimar tiempo de viaje (asumiendo 50 km/h promedio)
                    travel_times[i][j] = (distance_km / 50) * 60  # en minutos
        
        # Actualizar DataFrames usando índices consistentes con POIs (empezando desde '1')
        poi_indices = [str(i+1) for i in range(num_pois)]
        data['distances'] = pd.DataFrame(distances, index=poi_indices, columns=poi_indices)
        data['travel_times'] = pd.DataFrame(travel_times, index=poi_indices, columns=poi_indices)
        
        # También actualizar otras matrices para mantener consistencia de índices
        if 'costs' in data:
            costs_matrix = data['costs'].values
            if costs_matrix.shape[0] == num_pois and costs_matrix.shape[1] == num_pois:
                data['costs'] = pd.DataFrame(costs_matrix, index=poi_indices, columns=poi_indices)
        
        if 'co2_emission_cost' in data:
            co2_matrix = data['co2_emission_cost'].values  
            if co2_matrix.shape[0] == num_pois and co2_matrix.shape[1] == num_pois:
                data['co2_emission_cost'] = pd.DataFrame(co2_matrix, index=poi_indices, columns=poi_indices)
        
        if 'accident_risk' in data:
            risk_matrix = data['accident_risk'].values
            if risk_matrix.shape[0] == num_pois and risk_matrix.shape[1] == num_pois:
                data['accident_risk'] = pd.DataFrame(risk_matrix, index=poi_indices, columns=poi_indices)
        
        return data
    
    def _create_grupo_info_from_request(self, request) -> Dict[str, Any]:
        """Crear información del grupo turístico basada en la solicitud"""
        preferences = request.preferences
        constraints = request.constraints
        
        return {
            'tiempo_disponible': preferences.max_total_time if preferences.max_total_time > 0 else 720,
            'presupuesto': preferences.max_total_cost if preferences.max_total_cost > 0 else 500,
            'optimize_for': preferences.optimize_for if preferences.optimize_for else 'distance',
            'accessibility_required': preferences.accessibility_required,
            'start_time': constraints.start_time if constraints.start_time else '08:00',
            'lunch_break_required': constraints.lunch_break_required,
            'lunch_break_duration': constraints.lunch_break_duration if constraints.lunch_break_duration > 0 else 60,
            'min_pois_per_route': self.default_config['min_pois_per_route'],
            'max_pois_per_route': min(self.default_config['max_pois_per_route'], len(request.pois)),
            'origen': '1'  # Usar primer POI como origen (índice '1')
        }
    
    def _execute_mrl_amis_model(self, data: Dict[str, Any], request) -> Dict[str, Any]:
        """Ejecutar el modelo MRL-AMIS con los datos proporcionados"""
        logger.info("Ejecutando modelo MRL-AMIS...")
        
        grupo_info = data['grupo_info']
        tipo_turista = "nacional"  # Por defecto
        origen_poi = grupo_info['origen']
        
        # Parámetros del algoritmo
        num_work_packages = self.default_config['num_work_packages']
        max_iterations = self.default_config['max_iterations']
        maximize_objectives_list = self.default_config['maximize_objectives_list']
        ref_point_hypervolume = self.default_config['ref_point_hypervolume']
        
        # Generar Work Packages iniciales
        num_pois = len(data['pois'])
        wps_df = generar_work_packages(q=num_work_packages, d=num_pois)
        logger.info(f"Generados {len(wps_df)} Work Packages iniciales")
        
        # Decodificar y evaluar población inicial
        resultados_wp_iniciales = []
        for wp_name, wp_series in wps_df.iterrows():
            wp_vector = wp_series.values
            
            # Decodificar WP
            ruta_decodificada, is_feasible = decodificar_wp(
                wp_vector=wp_vector,
                pois_df=data["pois"],
                travel_times_df=data["travel_times"],
                grupo_info=grupo_info,
                origen=origen_poi
            )
            
            # Evaluar ruta
            metrics = evaluar_funciones_objetivo(ruta_decodificada, data, tipo_turista=tipo_turista)
            
            resultados_wp_iniciales.append({
                "wp_name": wp_name,
                "wp_original": wp_vector,
                "ruta_decodificada": ruta_decodificada,
                "objetivos": metrics,
                "is_feasible": is_feasible
            })
        
        # Inicializar agente RL
        rl_agent = QLearningAgent(
            state_space_size=self.rl_config['state_space_size'],
            action_space_size=self.rl_config['action_space_size'],
            learning_rate=self.rl_config['learning_rate'],
            discount_factor=self.rl_config['discount_factor'],
            epsilon=self.rl_config['epsilon_start'],
            epsilon_decay_rate=self.rl_config['epsilon_decay_rate'],
            min_epsilon=self.rl_config['min_epsilon']
        )
        
        # Bucle principal MRL-AMIS
        current_population_results = resultados_wp_iniciales.copy()
        hypervolume_history = []
        
        iteration_metrics = {
            'iteration': [],
            'pareto_front_size': [],
            'hypervolume': [],
            'arp': [],
            'spacing': []
        }
        
        logger.info(f"Iniciando bucle MRL-AMIS con {max_iterations} iteraciones...")
        
        for iteration in range(max_iterations):
            # Mostrar progreso cada 10 iteraciones
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteración {iteration + 1}/{max_iterations}")
            
            # Extraer objetivos de la población actual
            current_population_objectives_np_all = np.array([
                [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), 
                 obj_dict.get('co2_total', 0),
                 obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
                 obj_dict.get('riesgo_total', 0)]
                for sol in current_population_results
                for obj_dict in [sol.get('objetivos', {})]
                if obj_dict
            ])
            
            # Encontrar frente de Pareto actual
            current_feasible_indices = [i for i, sol in enumerate(current_population_results) if sol.get('is_feasible', False)]
            current_feasible_objectives_np = current_population_objectives_np_all[current_feasible_indices] if current_feasible_indices else np.array([])
            
            current_pareto_front_objectives = []
            if len(current_feasible_objectives_np) > 0:
                current_pareto_front_objectives, _ = find_pareto_front(current_feasible_objectives_np, maximize_objectives_list)
            
            # Generar nuevas soluciones usando Intelligence Boxes y RL
            generated_solutions_this_iteration = []
            
            for original_wp_result in current_population_results:
                # Obtener estado actual
                current_state = get_state(original_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)
                
                # Elegir acción usando RL
                action_index = rl_agent.choose_action(current_state)
                chosen_intelligence_box_func = self.intelligence_boxes[action_index]
                
                # Preparar parámetros para Intelligence Box
                original_wp_vector = original_wp_result.get('wp_original', np.array([]))
                kwargs = {
                    'grupo_info': grupo_info,
                    'pois_df': data["pois"],
                    'travel_times_df': data["travel_times"],
                    'data': data,
                    'tipo_turista': tipo_turista,
                    'maximize_objectives_list': maximize_objectives_list,
                    'current_pareto_front': current_pareto_front_objectives,
                    'current_population_results': current_population_results,
                    'current_pareto_front_indices': current_feasible_indices,
                    'origen': origen_poi
                }
                
                # Aplicar Intelligence Box
                modified_wp_vector = chosen_intelligence_box_func(original_wp_vector, **kwargs)
                
                # Decodificar y evaluar WP modificado
                decoded_route, is_feasible = decodificar_wp(
                    wp_vector=modified_wp_vector,
                    pois_df=data["pois"],
                    travel_times_df=data["travel_times"],
                    grupo_info=grupo_info,
                    origen=origen_poi
                )
                
                evaluated_metrics = evaluar_funciones_objetivo(decoded_route, data, tipo_turista)
                
                modified_wp_result = {
                    'wp_original': modified_wp_vector,
                    'ruta_decodificada': decoded_route,
                    'objetivos': evaluated_metrics,
                    'is_feasible': is_feasible
                }
                
                # Calcular recompensa y aprender
                reward = calculate_reward(original_wp_result, modified_wp_result, current_pareto_front_objectives, maximize_objectives_list)
                next_state = get_state(modified_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)
                rl_agent.learn(current_state, action_index, reward, next_state)
                
                generated_solutions_this_iteration.append(modified_wp_result)
            
            # Actualizar población
            current_population_results = update_population(
                current_population_results,
                generated_solutions_this_iteration,
                num_work_packages,
                maximize_objectives_list
            )
            
            # Calcular métricas de la iteración
            objectives_updated_pop_feasible = np.array([
                [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
                 obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
                 obj_dict.get('riesgo_total', 0)]
                for sol in current_population_results
                if sol.get('is_feasible', False)
                for obj_dict in [sol.get('objetivos', {})]
                if obj_dict
            ])
            
            iteration_hv = 0.0
            iteration_arp = 0.0
            iteration_spacing = 0.0
            iteration_pareto_size = 0
            
            if len(objectives_updated_pop_feasible) > 0:
                current_pareto_front_objectives_iter, pareto_indices = find_pareto_front(
                    objectives_updated_pop_feasible,
                    maximize_objectives_list
                )
                iteration_pareto_size = len(current_pareto_front_objectives_iter)
                
                if iteration_pareto_size > 0:
                    try:
                        iteration_hv = calculate_hypervolume(
                            current_pareto_front_objectives_iter,
                            ref_point_hypervolume,
                            maximize_objectives_list
                        )
                        hypervolume_history.append(iteration_hv)
                        
                        iteration_arp = calculate_average_ratio_pareto(
                            current_pareto_front_objectives_iter,
                            objectives_updated_pop_feasible,
                            maximize_objectives_list
                        )
                        
                        iteration_spacing = spacing_metric(
                            current_pareto_front_objectives_iter,
                            maximize_objectives_list
                        )
                    except Exception as e:
                        logger.warning(f"Error calculando métricas en iteración {iteration + 1}: {e}")
            
            # Almacenar métricas
            iteration_metrics['iteration'].append(iteration + 1)
            iteration_metrics['pareto_front_size'].append(iteration_pareto_size)
            iteration_metrics['hypervolume'].append(iteration_hv)
            iteration_metrics['arp'].append(iteration_arp)
            iteration_metrics['spacing'].append(iteration_spacing)
            
            # Decaimiento epsilon
            rl_agent.decay_epsilon_with_restart()
        
        # Obtener resultados finales
        final_population_objectives_feasible = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
             obj_dict.get('riesgo_total', 0)]
            for sol in current_population_results
            if sol.get('is_feasible', False)
            for obj_dict in [sol.get('objetivos', {})]
            if obj_dict
        ])
        
        final_pareto_front_objectives = []
        final_pareto_front_solutions = []
        
        if len(final_population_objectives_feasible) > 0:
            final_pareto_front_objectives, final_pareto_front_indices = find_pareto_front(
                final_population_objectives_feasible,
                maximize_objectives_list
            )
            
            final_feasible_solutions = [
                sol for sol in current_population_results
                if sol.get('is_feasible', False) and sol.get('objetivos') is not None
            ]
            
            final_pareto_front_solutions = [final_feasible_solutions[i] for i in final_pareto_front_indices]
        
        logger.info(f"Modelo MRL-AMIS completado. Frente de Pareto final: {len(final_pareto_front_solutions)} soluciones")
        
        return {
            'final_pareto_front_objectives': final_pareto_front_objectives,
            'final_pareto_front_solutions': final_pareto_front_solutions,
            'iteration_metrics': iteration_metrics,
            'hypervolume_history': hypervolume_history,
            'execution_time': 0  # Se calculará en el método llamador
        }
    
    def _convert_results_to_grpc_response(self, route_id: str, results: Dict[str, Any], execution_time: float) -> route_optimization_pb2.RouteOptimizationResponse:
        """Convertir resultados del modelo a respuesta gRPC"""
        try:
            final_pareto_front_solutions = results.get('final_pareto_front_solutions', [])
            
            if not final_pareto_front_solutions:
                return route_optimization_pb2.RouteOptimizationResponse(
                    route_id=route_id,
                    success=False,
                    message="No se encontraron soluciones factibles en el frente de Pareto"
                )
            
            # Seleccionar la mejor solución (primera del frente de Pareto)
            best_solution = final_pareto_front_solutions[0]
            best_route = best_solution['ruta_decodificada']
            best_objectives = best_solution['objetivos']
            
            # Crear secuencia optimizada de POIs
            optimized_sequence = []
            for i, poi_id in enumerate(best_route):
                if poi_id == '0':  # Saltar origen/destino si es necesario
                    continue
                    
                poi_index = int(poi_id) if poi_id.isdigit() else i
                
                optimized_poi = route_optimization_pb2.OptimizedPOI(
                    poi_id=poi_index,
                    poi_name=f"POI_{poi_index}",  # Se podría obtener el nombre real
                    visit_order=i,
                    arrival_time=f"{8 + i}:00",  # Tiempo estimado simplificado
                    departure_time=f"{8 + i + 1}:30",
                    estimated_visit_time=90,  # 1.5 horas por defecto
                    latitude=0.0,  # Se podría obtener de los datos originales
                    longitude=0.0
                )
                optimized_sequence.append(optimized_poi)
            
            # Crear resultados de optimización
            optimization_results = route_optimization_pb2.OptimizationResults(
                optimized_sequence=optimized_sequence,
                total_distance_km=best_objectives.get('distancia_total', 0) / 1000,  # Convertir a km si está en metros
                total_time_minutes=int(best_objectives.get('tiempo_total', 0)),
                total_cost=best_objectives.get('costo_total', 0),
                optimization_score=0.85,  # Score simplificado
                route_description=f"Ruta optimizada con {len(optimized_sequence)} POIs usando MRL-AMIS"
            )
            
            # Crear métricas de optimización
            iteration_metrics = results.get('iteration_metrics', {})
            final_hv = iteration_metrics.get('hypervolume', [0])[-1] if iteration_metrics.get('hypervolume') else 0
            final_arp = iteration_metrics.get('arp', [0])[-1] if iteration_metrics.get('arp') else 0
            final_spacing = iteration_metrics.get('spacing', [0])[-1] if iteration_metrics.get('spacing') else 0
            final_pareto_size = iteration_metrics.get('pareto_front_size', [0])[-1] if iteration_metrics.get('pareto_front_size') else 0
            total_iterations = len(iteration_metrics.get('iteration', [])) if iteration_metrics.get('iteration') else 0
            
            optimization_metrics = route_optimization_pb2.OptimizationMetrics(
                hypervolume=final_hv,
                arp=final_arp,
                spacing=final_spacing,
                pareto_front_size=final_pareto_size,
                total_iterations=total_iterations,
                execution_time_seconds=execution_time
            )
            
            return route_optimization_pb2.RouteOptimizationResponse(
                route_id=route_id,
                success=True,
                message=f"Optimización completada exitosamente. Encontradas {len(final_pareto_front_solutions)} soluciones en el frente de Pareto.",
                results=optimization_results,
                metrics=optimization_metrics
            )
            
        except Exception as e:
            logger.error(f"Error convirtiendo resultados a gRPC: {str(e)}", exc_info=True)
            return route_optimization_pb2.RouteOptimizationResponse(
                route_id=route_id,
                success=False,
                message=f"Error procesando resultados: {str(e)}"
            )

def serve():
    """Iniciar el servidor gRPC"""
    port = os.getenv('GRPC_PORT', '50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    route_optimization_pb2_grpc.add_RouteOptimizationServiceServicer_to_server(
        RouteOptimizationServicer(), server
    )
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Iniciando servidor gRPC en {listen_addr}")
    server.start()
    logger.info("Servidor gRPC iniciado correctamente")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Deteniendo servidor gRPC...")
        server.stop(grace=5)

if __name__ == '__main__':
    serve()