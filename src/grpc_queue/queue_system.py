"""Sistema de colas actualizado para el paquete grpc_queue."""

import queue
import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor

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


class QueueSystem:
    """Sistema de colas para gestionar trabajos de optimización MRL-AMIS con dataclasses."""
    
    def __init__(self, max_concurrent_jobs=1000, max_queue_size=50, use_multiprocessing=True):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        self.use_multiprocessing = False
        
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
        
        logger.info(f"QueueSystem inicializado: {max_concurrent_jobs} workers, "
                   f"multiprocessing={'activado' if use_multiprocessing else 'desactivado'}")
    
    def _start_workers(self):
        """Inicializar threads workers"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(i,), 
                daemon=True,
                name=f"DomainMRLAMISWorker-{i}"
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def submit_job(self, job_data: Dict[str, Any], request_data: Dict[str, Any]) -> str:
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
        logger.info(f"Worker {worker_id} iniciado con dataclasses")
        
        # Crear instancia del worker
        from grpc_queue.domain_worker import DomainMRLAMISWorker
        worker = DomainMRLAMISWorker()
        
        while not self.shutdown_event.is_set():
            try:
                # Obtener trabajo de la cola
                work_item = self.job_queue.get(timeout=1.0)
                job = work_item['job']
                job_data = work_item['job_data']
                request_data = work_item['request_data']
                
                # LOG 5: Datos recibidos por el worker
                logger.info("======================================================================")
                logger.info(f"PASO 5: Worker {worker_id} ha recogido el trabajo {job.job_id}")
                logger.info(f"  -> Datos del trabajo recibidos: {job_data}")
                logger.info(f"  -> Datos de la solicitud recibidos: {request_data}")
                
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
                    
                    
                    if self.use_multiprocessing and self.process_pool:
                        # Ejecutar en proceso separado
                        future = self.process_pool.submit(
                            worker.execute_optimization,
                            job_data, request_data, job.job_id, progress_callback
                        )
                        result = future.result(timeout=600)  # 10 minutos timeout
                    else:
                        # Ejecutar en thread actual
                        result = worker.execute_optimization(
                            job_data, request_data, job.job_id, progress_callback
                        )
                    
                    # Trabajo completado exitosamente
                    with self.lock:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        job.progress = 100.0
                        if job.started_at:
                            job.execution_time = (datetime.now() - job.started_at).total_seconds()
                        
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
_queue_system = None

def get_queue_system() -> QueueSystem:
    """Obtener instancia del sistema de colas (singleton)"""
    global _queue_system
    if _queue_system is None:
        _queue_system = QueueSystem(
            max_concurrent_jobs=1000,
            max_queue_size=50,
            use_multiprocessing=True
        )
    return _queue_system

def shutdown_queue_system():
    """Cerrar sistema de colas"""
    global _queue_system
    if _queue_system:
        _queue_system.shutdown()
        _queue_system = None