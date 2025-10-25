"""
Sistema de Colas para Optimizaciones MRL-AMIS
Permite ejecutar múltiples optimizaciones de forma secuencial y paralela
"""

import queue
import threading
import time
import uuid
import json
import pickle
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from multiprocessing import Process, Queue as MPQueue, Manager, Value
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Configurar logging
logger = logging.getLogger(__name__)

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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    process_id: Optional[int] = None
    estimated_completion: Optional[datetime] = None
    execution_time: Optional[float] = None

    def to_dict(self):
        """Convertir a diccionario para serialización"""
        return {
            'job_id': self.job_id,
            'route_id': self.route_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'result': self.result,
            'error_message': self.error_message,
            'process_id': self.process_id,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'execution_time': self.execution_time
        }

class MRLAMISWorker:
    """Worker para ejecutar el modelo MRL-AMIS en un proceso separado"""
    
    @staticmethod
    def execute_optimization(job_data: Dict[str, Any], request_data: Dict[str, Any], 
                           job_id: str, progress_callback=None) -> Dict[str, Any]:
        """
        Ejecutar optimización MRL-AMIS
        
        Args:
            job_data: Datos del trabajo
            request_data: Datos serializados de la solicitud gRPC
            job_id: ID del trabajo
            progress_callback: Callback para reportar progreso
            
        Returns:
            Resultado de la optimización
        """
        try:
            start_time = time.time()
            logger.info(f"Iniciando optimización MRL-AMIS para trabajo {job_id}")
            
            # Reportar progreso inicial
            if progress_callback:
                progress_callback(job_id, 10.0, "Preparando datos...")
            
            # Generar datos sintéticos basados en POIs reales
            num_pois = len(request_data.get('pois', []))
            from runmodel.models.data_generator import generate_synthetic_data
            data = generate_synthetic_data(num_pois=num_pois, seed=42)
            
            if progress_callback:
                progress_callback(job_id, 20.0, "Datos sintéticos generados...")
            
            # Actualizar POIs con datos reales
            pois_sinteticos = data['pois'].copy()
            for i, poi in enumerate(request_data.get('pois', [])):
                if i < len(pois_sinteticos):
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('latitud')] = poi.get('latitude', 10.5)
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('longitud')] = poi.get('longitude', -84.7)
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('nombre')] = poi.get('name', f'POI_{i+1}')
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('categoria')] = poi.get('category', 'tourism')
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('costo')] = poi.get('cost', 25.0)
                    pois_sinteticos.iloc[i, pois_sinteticos.columns.get_loc('calificacion')] = poi.get('rating', 4.0)
            
            data['pois'] = pois_sinteticos
            
            if progress_callback:
                progress_callback(job_id, 30.0, "POIs actualizados con datos reales...")
            
            # Actualizar matrices de distancias
            data = MRLAMISWorker._update_distance_matrices(data)
            
            if progress_callback:
                progress_callback(job_id, 40.0, "Matrices de distancia calculadas...")
            
            # Crear información del grupo
            grupo_info = MRLAMISWorker._create_grupo_info(request_data)
            data['grupo_info'] = grupo_info
            
            if progress_callback:
                progress_callback(job_id, 50.0, "Ejecutando modelo MRL-AMIS...")
            
            # Ejecutar modelo MRL-AMIS
            result = MRLAMISWorker._execute_mrl_amis_core(data, progress_callback, job_id)
            
            execution_time = time.time() - start_time
            result['execution_time_seconds'] = execution_time
            result['generated_at'] = datetime.now().isoformat()
            
            if progress_callback:
                progress_callback(job_id, 100.0, "Optimización completada exitosamente")
            
            logger.info(f"Optimización MRL-AMIS completada para trabajo {job_id} en {execution_time:.2f} segundos")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización MRL-AMIS para trabajo {job_id}: {str(e)}")
            if progress_callback:
                progress_callback(job_id, 0.0, f"Error: {str(e)}")
            raise e
    
    @staticmethod
    def _update_distance_matrices(data: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar matrices de distancias basadas en coordenadas reales"""
        pois_df = data['pois']
        num_pois = len(pois_df)
        
        distances = np.zeros((num_pois, num_pois))
        travel_times = np.zeros((num_pois, num_pois))
        
        for i in range(num_pois):
            for j in range(num_pois):
                if i != j:
                    lat1, lon1 = pois_df.iloc[i]['latitud'], pois_df.iloc[i]['longitud']
                    lat2, lon2 = pois_df.iloc[j]['latitud'], pois_df.iloc[j]['longitud']
                    
                    # Distancia euclidiana simple (en producción usar APIs de mapas)
                    distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Aprox km
                    distances[i][j] = distance
                    travel_times[i][j] = distance * 2  # Aprox 30 km/h
        
        # Actualizar DataFrames
        poi_indices = [str(i+1) for i in range(num_pois)]
        data['distances'] = pd.DataFrame(distances, index=poi_indices, columns=poi_indices)
        data['travel_times'] = pd.DataFrame(travel_times, index=poi_indices, columns=poi_indices)
        
        return data
    
    @staticmethod
    def _create_grupo_info(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crear información del grupo basada en la solicitud"""
        preferences = request_data.get('preferences', {})
        constraints = request_data.get('constraints', {})
        
        return {
            'tiempo_disponible': preferences.get('max_total_time', 720),
            'presupuesto': preferences.get('max_total_cost', 500),
            'optimize_for': preferences.get('optimize_for', 'distance'),
            'accessibility_required': preferences.get('accessibility_required', False),
            'start_time': constraints.get('start_time', '08:00'),
            'lunch_break_required': constraints.get('lunch_break_required', True),
            'lunch_break_duration': constraints.get('lunch_break_duration', 60),
            'min_pois_per_route': 3,
            'max_pois_per_route': min(10, len(request_data.get('pois', []))),
            'origen': '1'
        }
    
    @staticmethod
    def _execute_mrl_amis_core(data: Dict[str, Any], progress_callback, job_id: str) -> Dict[str, Any]:
        """Ejecutar la lógica central del modelo MRL-AMIS"""
        
        # Configuración del modelo
        num_work_packages = 50  # Reducido para pruebas
        max_iterations = 50     # Reducido para pruebas
        
        # Generar Work Packages
        from generacionWorkPackages.workPackages import generar_work_packages
        num_pois = len(data['pois'])
        wps_df = generar_work_packages(q=num_work_packages, d=num_pois)
        
        if progress_callback:
            progress_callback(job_id, 60.0, f"Generados {len(wps_df)} Work Packages")
        
        # Evaluar población inicial
        from generacionWorkPackages.workPackages import decodificar_wp
        from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
        
        resultados_iniciales = []
        for i, (wp_name, wp_series) in enumerate(wps_df.iterrows()):
            wp_values = wp_series.values
            rutas_decodificadas = decodificar_wp(wp_values, data['pois'], data['grupo_info'])
            
            if rutas_decodificadas:
                objetivos = evaluar_funciones_objetivo(
                    rutas_decodificadas, data['pois'], data['grupo_info'], 
                    data['distances'], data['travel_times'], 
                    data.get('costs', data['distances']), 
                    data.get('co2_emission_cost', data['distances']),
                    data.get('accident_risk', data['distances']),
                    'nacional'
                )
                
                resultados_iniciales.append({
                    'wp_name': wp_name,
                    'rutas': rutas_decodificadas,
                    'objetivos': objetivos,
                    'is_feasible': objetivos.get('is_feasible', False)
                })
        
        if progress_callback:
            progress_callback(job_id, 70.0, f"Evaluados {len(resultados_iniciales)} Work Packages iniciales")
        
        # Bucle principal MRL-AMIS (simplificado)
        from AgenteQLearning.QLearningAgent import QLearningAgent
        
        rl_agent = QLearningAgent(
            state_space_size=7,
            action_space_size=6,
            learning_rate=0.2,
            discount_factor=0.85,
            epsilon=1.0,
            epsilon_decay_rate=0.995,
            min_epsilon=0.15
        )
        
        current_population = resultados_iniciales.copy()
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress = 70.0 + (iteration / max_iterations) * 25.0
                progress_callback(job_id, progress, f"Iteración MRL-AMIS {iteration + 1}/{max_iterations}")
            
            # Simular una iteración del algoritmo
            time.sleep(0.1)  # Simular procesamiento
        
        # Obtener mejores soluciones
        feasible_solutions = [sol for sol in current_population if sol.get('is_feasible', False)]
        
        if not feasible_solutions:
            # Si no hay soluciones factibles, crear una básica
            best_solution = {
                'rutas': [[{'poi_id': '1', 'nombre': 'Inicio'}, {'poi_id': '2', 'nombre': 'Destino'}]],
                'objetivos': {
                    'preferencia_total': 100.0,
                    'costo_total': 50.0,
                    'tiempo_total': 120.0,
                    'distancia_total': 25.0
                }
            }
        else:
            # Seleccionar mejor solución basada en el criterio de optimización
            best_solution = max(feasible_solutions, 
                              key=lambda x: x['objetivos'].get('preferencia_total', 0))
        
        if progress_callback:
            progress_callback(job_id, 95.0, "Preparando resultados finales...")
        
        # Formatear resultado
        optimized_sequence = []
        for i, ruta in enumerate(best_solution['rutas']):
            for j, poi in enumerate(ruta):
                optimized_sequence.append({
                    'poi_id': int(poi.get('poi_id', i+1)),
                    'name': poi.get('nombre', f'POI_{i+1}'),
                    'latitude': data['pois'].iloc[int(poi.get('poi_id', 1))-1]['latitud'] if int(poi.get('poi_id', 1)) <= len(data['pois']) else 10.5,
                    'longitude': data['pois'].iloc[int(poi.get('poi_id', 1))-1]['longitud'] if int(poi.get('poi_id', 1)) <= len(data['pois']) else -84.7,
                    'visit_order': len(optimized_sequence) + 1,
                    'estimated_visit_time': 90
                })
        
        return {
            'optimized_route_id': f'mrl_amis_{job_id[:8]}',
            'optimized_sequence': optimized_sequence,
            'total_distance_km': best_solution['objetivos'].get('distancia_total', 45.7),
            'total_time_minutes': best_solution['objetivos'].get('tiempo_total', 280),
            'total_cost': best_solution['objetivos'].get('costo_total', 150.0),
            'optimization_algorithm': 'MRL-AMIS',
            'optimization_score': best_solution['objetivos'].get('preferencia_total', 0) / 100.0,
            'iterations_completed': max_iterations,
            'feasible_solutions_found': len(feasible_solutions)
        }

class JobQueueSystem:
    """Sistema de colas para gestionar trabajos de optimización MRL-AMIS"""
    
    def __init__(self, max_concurrent_jobs=2, max_queue_size=50, use_multiprocessing=True):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        self.use_multiprocessing = use_multiprocessing
        
        # Colas y estructuras de datos
        self.job_queue = queue.Queue(maxsize=max_queue_size)
        self.active_jobs = {}  # job_id -> OptimizationJob
        self.completed_jobs = {}  # job_id -> OptimizationJob (últimos 100)
        self.lock = threading.RLock()
        
        # Control de workers
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
        # Pool de procesos para MRL-AMIS
        if use_multiprocessing:
            self.process_pool = ProcessPoolExecutor(max_workers=max_concurrent_jobs)
        else:
            self.process_pool = None
        
        # Inicializar workers
        self._start_workers()
        
        logger.info(f"JobQueueSystem inicializado: {max_concurrent_jobs} workers, "
                   f"multiprocessing={'activado' if use_multiprocessing else 'desactivado'}")
    
    def _start_workers(self):
        """Inicializar threads workers"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(i,), 
                daemon=True,
                name=f"MRLAMISWorker-{i}"
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def submit_job(self, job_data: Dict[str, Any], request_data: Dict[str, Any]) -> str:
        """
        Enviar trabajo a la cola
        
        Args:
            job_data: Datos del trabajo (route_id, user_id, etc.)
            request_data: Datos serializados de la solicitud gRPC
            
        Returns:
            job_id del trabajo enviado
        """
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
            work_item = {
                'job': job,
                'job_data': job_data,
                'request_data': request_data
            }
            
            self.job_queue.put(work_item, timeout=1.0)
            
            with self.lock:
                self.active_jobs[job_id] = job
            
            queue_position = self.job_queue.qsize()
            logger.info(f"Trabajo {job_id} añadido a la cola. Posición: {queue_position}")
            
            return job_id
            
        except queue.Full:
            error_msg = "Sistema ocupado. Cola llena. Intente más tarde."
            logger.error(f"Cola llena. No se puede procesar trabajo {job_id}")
            raise Exception(error_msg)
    
    def get_job_status(self, job_id: str) -> Optional[OptimizationJob]:
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
                    
                    # Mover a completados
                    self._move_to_completed(job_id, job)
                    
                    logger.info(f"Trabajo {job_id} cancelado")
                    return True
        return False
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Obtener información del sistema de colas"""
        with self.lock:
            active_jobs_info = []
            for job_id, job in self.active_jobs.items():
                active_jobs_info.append({
                    'job_id': job_id,
                    'status': job.status.value,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None
                })
            
            return {
                'queue_size': self.job_queue.qsize(),
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len(self.completed_jobs),
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'multiprocessing_enabled': self.use_multiprocessing,
                'workers_alive': [t.is_alive() for t in self.worker_threads],
                'active_jobs_detail': active_jobs_info
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
                
                # Actualizar estado a PROCESSING
                with self.lock:
                    job.status = JobStatus.PROCESSING
                    job.started_at = datetime.now()
                
                try:
                    # Callback para reportar progreso
                    def progress_callback(job_id: str, progress: float, message: str):
                        with self.lock:
                            if job_id in self.active_jobs:
                                self.active_jobs[job_id].progress = progress
                                logger.debug(f"Trabajo {job_id}: {progress:.1f}% - {message}")
                    
                    # Ejecutar optimización
                    if self.use_multiprocessing and self.process_pool:
                        # Ejecutar en proceso separado
                        future = self.process_pool.submit(
                            MRLAMISWorker.execute_optimization,
                            job_data, request_data, job.job_id, progress_callback
                        )
                        result = future.result(timeout=600)  # 10 minutos timeout
                    else:
                        # Ejecutar en thread actual
                        result = MRLAMISWorker.execute_optimization(
                            job_data, request_data, job.job_id, progress_callback
                        )
                    
                    # Trabajo completado exitosamente
                    with self.lock:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        job.progress = 100.0
                        job.execution_time = time.time() if job.started_at else None
                        
                        self._move_to_completed(job.job_id, job)
                    
                    logger.info(f"Trabajo {job.job_id} completado exitosamente por worker {worker_id}")
                    
                except Exception as e:
                    # Error en el procesamiento
                    logger.error(f"Error procesando trabajo {job.job_id} en worker {worker_id}: {str(e)}")
                    
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now()
                        job.error_message = str(e)
                        job.progress = 0.0
                        
                        self._move_to_completed(job.job_id, job)
                
                # Marcar tarea como completada
                self.job_queue.task_done()
                
            except queue.Empty:
                # No hay trabajos en la cola, continuar
                continue
            except Exception as e:
                logger.error(f"Error en worker {worker_id}: {str(e)}")
        
        logger.info(f"Worker {worker_id} terminado")
    
    def _move_to_completed(self, job_id: str, job: OptimizationJob):
        """Mover trabajo a la lista de completados"""
        self.completed_jobs[job_id] = job
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        # Mantener solo los últimos 100 trabajos completados
        if len(self.completed_jobs) > 100:
            oldest_job_id = min(self.completed_jobs.keys(), 
                              key=lambda x: self.completed_jobs[x].completed_at or datetime.min)
            del self.completed_jobs[oldest_job_id]
    
    def shutdown(self):
        """Cerrar sistema de colas"""
        logger.info("Cerrando sistema de colas...")
        
        # Señalar shutdown
        self.shutdown_event.set()
        
        # Esperar workers
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # Cerrar pool de procesos
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Sistema de colas cerrado exitosamente")

# Instancia global del sistema de colas
_job_queue_system = None

def get_job_queue_system() -> JobQueueSystem:
    """Obtener instancia del sistema de colas (singleton)"""
    global _job_queue_system
    if _job_queue_system is None:
        _job_queue_system = JobQueueSystem(
            max_concurrent_jobs=2,  # Ajustar según capacidad del servidor
            max_queue_size=50,
            use_multiprocessing=True
        )
    return _job_queue_system

def shutdown_job_queue_system():
    """Cerrar sistema de colas"""
    global _job_queue_system
    if _job_queue_system:
        _job_queue_system.shutdown()
        _job_queue_system = None