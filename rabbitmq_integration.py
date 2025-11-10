"""
Integración con RabbitMQ para el Sistema de Optimización de Rutas
Permite distribuir trabajos a través de múltiples instancias del servicio
"""

import pika
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

class RabbitMQJobDistributor:
    """
    Distribuidor de trabajos usando RabbitMQ
    Permite escalar horizontalmente el procesamiento de optimizaciones
    """
    
    def __init__(self, 
                 rabbitmq_host='localhost',
                 rabbitmq_port=5672,
                 rabbitmq_user='guest',
                 rabbitmq_password='guest',
                 queue_name='route_optimization_jobs',
                 result_queue_name='route_optimization_results'):
        
        self.host = rabbitmq_host
        self.port = rabbitmq_port
        self.user = rabbitmq_user
        self.password = rabbitmq_password
        self.queue_name = queue_name
        self.result_queue_name = result_queue_name
        
        self.connection = None
        self.channel = None
        self.result_channel = None
        self.is_connected = False
        
        # Callbacks para manejo de trabajos
        self.job_processor: Optional[Callable] = None
        self.result_handler: Optional[Callable] = None
        
        # Control de hilos
        self.consumer_thread = None
        self.result_consumer_thread = None
        self.shutdown_event = threading.Event()
        
        self._connect()
    
    def _connect(self):
        """Establecer conexión con RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.result_channel = self.connection.channel()
            
            # Declarar colas
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            self.result_channel.queue_declare(queue=self.result_queue_name, durable=True)
            
            # Configurar QoS para distribuir trabajos equitativamente
            self.channel.basic_qos(prefetch_count=1)
            
            self.is_connected = True
            logger.info(f"Conectado a RabbitMQ en {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Error conectando a RabbitMQ: {str(e)}")
            self.is_connected = False
            raise
    
    def publish_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Publicar trabajo en la cola de RabbitMQ
        
        Args:
            job_data: Datos del trabajo de optimización
            
        Returns:
            True si se publicó exitosamente
        """
        if not self.is_connected:
            logger.error("No hay conexión con RabbitMQ")
            return False
        
        try:
            # Añadir timestamp y metadatos
            job_message = {
                'job_data': job_data,
                'timestamp': datetime.utcnow().isoformat(),
                'sender': 'route-optimizer-service',
                'priority': job_data.get('priority', 5)  # 1-10, 10 es más alta prioridad
            }
            
            # Publicar mensaje
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=json.dumps(job_message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Hacer mensaje persistente
                    priority=job_message['priority']
                )
            )
            
            logger.info(f"Trabajo publicado en RabbitMQ: {job_data.get('job_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error publicando trabajo en RabbitMQ: {str(e)}")
            return False
    
    def start_consumer(self, job_processor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Iniciar consumidor de trabajos
        
        Args:
            job_processor: Función que procesa los trabajos recibidos
        """
        self.job_processor = job_processor
        
        def consumer_loop():
            logger.info("Iniciando consumidor de trabajos RabbitMQ")
            
            def callback(ch, method, properties, body):
                try:
                    # Deserializar mensaje
                    message = json.loads(body.decode('utf-8'))
                    job_data = message.get('job_data', {})
                    
                    logger.info(f"Procesando trabajo desde RabbitMQ: {job_data.get('job_id', 'unknown')}")
                    
                    # Procesar trabajo
                    result = self.job_processor(job_data)
                    
                    # Publicar resultado
                    if result:
                        self.publish_result(result)
                    
                    # Confirmar procesamiento
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"Error procesando trabajo desde RabbitMQ: {str(e)}")
                    # Rechazar mensaje sin reenvío
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # Configurar consumidor
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=callback
            )
            
            # Iniciar consumo
            while not self.shutdown_event.is_set():
                try:
                    self.connection.process_data_events(time_limit=1.0)
                except Exception as e:
                    logger.error(f"Error en bucle de consumo: {str(e)}")
                    time.sleep(1.0)
            
            logger.info("Consumidor de trabajos RabbitMQ terminado")
        
        self.consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
        self.consumer_thread.start()
    
    def publish_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Publicar resultado de trabajo procesado
        
        Args:
            result_data: Resultado del procesamiento
            
        Returns:
            True si se publicó exitosamente
        """
        if not self.is_connected:
            logger.error("No hay conexión con RabbitMQ para publicar resultado")
            return False
        
        try:
            result_message = {
                'result_data': result_data,
                'timestamp': datetime.utcnow().isoformat(),
                'sender': 'mrl-amis-service'
            }
            
            self.result_channel.basic_publish(
                exchange='',
                routing_key=self.result_queue_name,
                body=json.dumps(result_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            logger.info(f"Resultado publicado en RabbitMQ: {result_data.get('job_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error publicando resultado en RabbitMQ: {str(e)}")
            return False
    
    def start_result_consumer(self, result_handler: Callable[[Dict[str, Any]], None]):
        """
        Iniciar consumidor de resultados
        
        Args:
            result_handler: Función que maneja los resultados recibidos
        """
        self.result_handler = result_handler
        
        def result_consumer_loop():
            logger.info("Iniciando consumidor de resultados RabbitMQ")
            
            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body.decode('utf-8'))
                    result_data = message.get('result_data', {})
                    
                    logger.info(f"Resultado recibido desde RabbitMQ: {result_data.get('job_id', 'unknown')}")
                    
                    # Manejar resultado
                    self.result_handler(result_data)
                    
                    # Confirmar procesamiento
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"Error procesando resultado desde RabbitMQ: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            self.result_channel.basic_consume(
                queue=self.result_queue_name,
                on_message_callback=callback
            )
            
            while not self.shutdown_event.is_set():
                try:
                    self.connection.process_data_events(time_limit=1.0)
                except Exception as e:
                    logger.error(f"Error en bucle de consumo de resultados: {str(e)}")
                    time.sleep(1.0)
            
            logger.info("Consumidor de resultados RabbitMQ terminado")
        
        self.result_consumer_thread = threading.Thread(target=result_consumer_loop, daemon=True)
        self.result_consumer_thread.start()
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de las colas"""
        if not self.is_connected:
            return {'jobs_queue': 0, 'results_queue': 0}
        
        try:
            jobs_queue = self.channel.queue_declare(queue=self.queue_name, passive=True)
            results_queue = self.result_channel.queue_declare(queue=self.result_queue_name, passive=True)
            
            return {
                'jobs_queue': jobs_queue.method.message_count,
                'results_queue': results_queue.method.message_count
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de cola: {str(e)}")
            return {'jobs_queue': 0, 'results_queue': 0}
    
    def shutdown(self):
        """Cerrar conexión y consumidores"""
        logger.info("Cerrando conexión RabbitMQ...")
        
        self.shutdown_event.set()
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        if self.result_consumer_thread and self.result_consumer_thread.is_alive():
            self.result_consumer_thread.join(timeout=5.0)
        
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        
        self.is_connected = False
        logger.info("Conexión RabbitMQ cerrada")

class HybridJobManager:
    """
    Gestor híbrido que combina colas locales con RabbitMQ
    para máxima flexibilidad y escalabilidad
    """
    
    def __init__(self, 
                 use_rabbitmq=True,
                 rabbitmq_config=None,
                 local_queue_config=None):
        
        self.use_rabbitmq = use_rabbitmq
        self.local_queue = None
        self.rabbitmq_distributor = None
        
        # Inicializar cola local
        if local_queue_config:
            from job_queue_system import JobQueueSystem
            self.local_queue = JobQueueSystem(**local_queue_config)
        
        # Inicializar RabbitMQ si está habilitado
        if use_rabbitmq and rabbitmq_config:
            try:
                self.rabbitmq_distributor = RabbitMQJobDistributor(**rabbitmq_config)
                logger.info("Sistema híbrido: RabbitMQ + Cola local habilitado")
            except Exception as e:
                logger.warning(f"No se pudo conectar a RabbitMQ, usando solo cola local: {str(e)}")
                self.use_rabbitmq = False
        
        if not self.use_rabbitmq:
            logger.info("Sistema híbrido: Solo cola local habilitada")
    
    def submit_job(self, job_data: Dict[str, Any], request_data: Dict[str, Any]) -> str:
        """
        Enviar trabajo usando la estrategia más apropiada
        
        Estrategia:
        1. Si RabbitMQ está disponible y la cola local está llena -> RabbitMQ
        2. Si RabbitMQ no está disponible o cola local tiene capacidad -> Cola local
        """
        
        # Intentar cola local primero
        if self.local_queue:
            try:
                queue_info = self.local_queue.get_queue_info()
                local_capacity = queue_info['queue_size'] < (queue_info.get('max_queue_size', 50) * 0.8)
                
                if local_capacity:
                    job_id = self.local_queue.submit_job(job_data, request_data)
                    logger.info(f"Trabajo {job_id} asignado a cola local")
                    return job_id
                
            except Exception as e:
                logger.warning(f"Error en cola local, intentando RabbitMQ: {str(e)}")
        
        # Intentar RabbitMQ si cola local está llena o no disponible
        if self.use_rabbitmq and self.rabbitmq_distributor:
            try:
                # Crear job_id para tracking
                import uuid
                job_id = str(uuid.uuid4())
                
                job_data_with_id = {**job_data, 'job_id': job_id}
                distributed_job = {
                    'job_data': job_data_with_id,
                    'request_data': request_data
                }
                
                success = self.rabbitmq_distributor.publish_job(distributed_job)
                if success:
                    logger.info(f"Trabajo {job_id} distribuido via RabbitMQ")
                    return job_id
                
            except Exception as e:
                logger.error(f"Error distribuyendo trabajo via RabbitMQ: {str(e)}")
        
        # Si todo falla, lanzar excepción
        raise Exception("No se pudo procesar el trabajo: sistema ocupado")
    
    def get_job_status(self, job_id: str):
        """Obtener estado del trabajo (buscar en cola local primero)"""
        if self.local_queue:
            job = self.local_queue.get_job_status(job_id)
            if job:
                return job
        
        # Para trabajos distribuidos, necesitaríamos un sistema de tracking
        # Por simplicidad, retornamos None si no se encuentra localmente
        return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema completo"""
        info = {
            'hybrid_mode': True,
            'rabbitmq_enabled': self.use_rabbitmq,
            'local_queue_info': None,
            'rabbitmq_stats': None
        }
        
        if self.local_queue:
            info['local_queue_info'] = self.local_queue.get_queue_info()
        
        if self.use_rabbitmq and self.rabbitmq_distributor:
            info['rabbitmq_stats'] = self.rabbitmq_distributor.get_queue_stats()
        
        return info
    
    def shutdown(self):
        """Cerrar todos los sistemas"""
        if self.local_queue:
            self.local_queue.shutdown()
        
        if self.rabbitmq_distributor:
            self.rabbitmq_distributor.shutdown()

# Instancia global del gestor híbrido
_hybrid_manager = None

def get_hybrid_job_manager() -> HybridJobManager:
    """Obtener instancia del gestor híbrido (singleton)"""
    global _hybrid_manager
    if _hybrid_manager is None:
        # Configuración por defecto
        rabbitmq_config = {
            'rabbitmq_host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'rabbitmq_port': int(os.getenv('RABBITMQ_PORT', '5672')),
            'rabbitmq_user': os.getenv('RABBITMQ_USER', 'guest'),
            'rabbitmq_password': os.getenv('RABBITMQ_PASSWORD', 'guest')
        }
        
        local_queue_config = {
            'max_concurrent_jobs': int(os.getenv('MAX_CONCURRENT_JOBS', '1000')),
            'max_queue_size': int(os.getenv('MAX_QUEUE_SIZE', '50')),
            'use_multiprocessing': os.getenv('USE_MULTIPROCESSING', 'true').lower() == 'true'
        }
        
        _hybrid_manager = HybridJobManager(
            use_rabbitmq=os.getenv('USE_RABBITMQ', 'true').lower() == 'true',
            rabbitmq_config=rabbitmq_config,
            local_queue_config=local_queue_config
        )
    
    return _hybrid_manager

def shutdown_hybrid_manager():
    """Cerrar gestor híbrido"""
    global _hybrid_manager
    if _hybrid_manager:
        _hybrid_manager.shutdown()
        _hybrid_manager = None