import grpc
from concurrent import futures
import time
import logging
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar los protobuf generados (se generarán después)
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
    import route_optimization_pb2
    import route_optimization_pb2_grpc
except ImportError:
    print("Error: Los archivos protobuf no han sido generados. Ejecuta:")
    print("./generate_proto.sh")
    print("O manualmente:")
    print("python -m grpc_tools.protoc -I./protos --python_out=./generated --grpc_python_out=./generated ./protos/route_optimization.proto")
    sys.exit(1)

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

class RouteOptimizationServicer(route_optimization_pb2_grpc.RouteOptimizationServiceServicer):
    
    def __init__(self):
        """Inicializar el servicio de optimización de rutas con sistema de colas"""
        logger.info("Inicializando RouteOptimizationService...")
        
        # Importar sistema de colas
        from job_queue_system import get_job_queue_system
        self.job_queue = get_job_queue_system()
        
        # Configuración predeterminada del modelo MRL-AMIS
        self.default_config = {
            'num_work_packages': 100,
            'max_iterations': 100,
            'num_pois': 15,
            'max_pois_per_route': 10,
            'min_pois_per_route': 3,
            'num_routes_per_wp': 3,
            'max_duration_per_route': 720,
            'maximize_objectives_list': [True, False, False, True, False],  # [preferencia, costo, co2, sustentabilidad, riesgo]
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
        
        logger.info("RouteOptimizationService inicializado correctamente con sistema de colas")
    
    def OptimizeRoute(self, request, context):
        """Método principal para optimizar rutas usando MRL-AMIS con sistema de colas"""
        try:
            logger.info(f"=== NUEVA SOLICITUD DE OPTIMIZACIÓN ===")
            logger.info(f"Route ID: {request.route_id}")
            logger.info(f"User ID: {request.user_id}")
            logger.info(f"Número de POIs: {len(request.pois)}")
            
            # Validar solicitud
            if not request.pois:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Al menos un POI es requerido')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            if len(request.pois) > 20:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Máximo 20 POIs permitidos')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            # Preparar datos del trabajo
            job_data = {
                'route_id': request.route_id or f'route_{int(time.time())}',
                'user_id': request.user_id or 'anonymous'
            }
            
            # Serializar datos de la solicitud
            request_data = self._serialize_grpc_request(request)
            
            # Enviar trabajo a la cola
            job_id = self.job_queue.submit_job(job_data, request_data)
            
            logger.info(f"Trabajo enviado a la cola con ID: {job_id}")
            
            # Crear respuesta con información del trabajo
            response = route_optimization_pb2.RouteOptimizationResponse()
            response.route_id = job_data['route_id']
            response.job_id = job_id
            response.status = "QUEUED"
            response.message = "Solicitud de optimización recibida y en cola para procesamiento"
            
            # Añadir información de la cola
            queue_info = self.job_queue.get_queue_info()
            response.queue_position = queue_info['queue_size']
            response.estimated_wait_time_minutes = queue_info['queue_size'] * 5  # Estimación simple
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando solicitud de optimización: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.RouteOptimizationResponse()
    
    def GetJobStatus(self, request, context):
        """Obtener estado de un trabajo de optimización"""
        try:
            job_id = request.job_id
            logger.debug(f"Consultando estado del trabajo: {job_id}")
            
            job = self.job_queue.get_job_status(job_id)
            
            if not job:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Trabajo {job_id} no encontrado')
                return route_optimization_pb2.JobStatusResponse()
            
            # Crear respuesta de estado
            response = route_optimization_pb2.JobStatusResponse()
            response.job_id = job.job_id
            response.route_id = job.route_id
            response.status = job.status.value
            response.progress = job.progress
            response.created_at = job.created_at.isoformat()
            
            if job.started_at:
                response.started_at = job.started_at.isoformat()
            if job.completed_at:
                response.completed_at = job.completed_at.isoformat()
            if job.estimated_completion:
                response.estimated_completion_time = job.estimated_completion.isoformat()
            if job.error_message:
                response.error_message = job.error_message
            
            # Si está completado, incluir resultados
            if job.status.value == "COMPLETED" and job.result:
                response.has_result = True
                # Los resultados se obtienen por separado con GetJobResult
            
            return response
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del trabajo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.JobStatusResponse()
    
    def GetJobResult(self, request, context):
        """Obtener resultado de un trabajo completado"""
        try:
            job_id = request.job_id
            logger.info(f"Solicitando resultado del trabajo: {job_id}")
            
            job = self.job_queue.get_job_status(job_id)
            
            if not job:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'Trabajo {job_id} no encontrado')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            if job.status.value != "COMPLETED":
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f'Trabajo {job_id} no está completado (estado: {job.status.value})')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            if not job.result:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f'No hay resultado disponible para el trabajo {job_id}')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            # Convertir resultado a respuesta gRPC
            return self._convert_results_to_grpc_response(
                job.route_id, job.result, job.execution_time or 0.0, job_id
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo resultado del trabajo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.RouteOptimizationResponse()
    
    def CancelJob(self, request, context):
        """Cancelar un trabajo de optimización"""
        try:
            job_id = request.job_id
            logger.info(f"Solicitando cancelación del trabajo: {job_id}")
            
            success = self.job_queue.cancel_job(job_id)
            
            response = route_optimization_pb2.CancelJobResponse()
            response.job_id = job_id
            response.success = success
            
            if success:
                response.message = f"Trabajo {job_id} cancelado exitosamente"
            else:
                response.message = f"No se pudo cancelar el trabajo {job_id} (puede que ya esté completado o no exista)"
            
            return response
            
        except Exception as e:
            logger.error(f"Error cancelando trabajo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.CancelJobResponse()
    
    def GetQueueInfo(self, request, context):
        """Obtener información del sistema de colas"""
        try:
            queue_info = self.job_queue.get_queue_info()
            
            response = route_optimization_pb2.QueueInfoResponse()
            response.queue_size = queue_info['queue_size']
            response.active_jobs = queue_info['active_jobs']
            response.completed_jobs = queue_info['completed_jobs']
            response.max_concurrent_jobs = queue_info['max_concurrent_jobs']
            response.multiprocessing_enabled = queue_info['multiprocessing_enabled']
            
            # Información detallada de trabajos activos
            for job_info in queue_info['active_jobs_detail']:
                job_detail = response.active_jobs_detail.add()
                job_detail.job_id = job_info['job_id']
                job_detail.status = job_info['status']
                job_detail.progress = job_info['progress']
                job_detail.created_at = job_info['created_at']
                if job_info['started_at']:
                    job_detail.started_at = job_info['started_at']
            
            return response
            
        except Exception as e:
            logger.error(f"Error obteniendo información de la cola: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.QueueInfoResponse()
    
    def HealthCheck(self, request, context):
        """Verificar el estado del servicio"""
        try:
            queue_info = self.job_queue.get_queue_info()
            
            return route_optimization_pb2.HealthResponse(
                is_healthy=True,
                status=f"Service running - Queue: {queue_info['queue_size']} waiting, {queue_info['active_jobs']} active",
                version="2.0.0",
                queue_size=queue_info['queue_size'],
                active_jobs=queue_info['active_jobs']
            )
        except Exception as e:
            logger.error(f"Error en health check: {str(e)}")
            return route_optimization_pb2.HealthResponse(
                is_healthy=False,
                status=f"Service error: {str(e)}",
                version="2.0.0"
            )
    
    def _serialize_grpc_request(self, request) -> Dict[str, Any]:
        """Serializar solicitud gRPC para procesamiento en cola"""
        return {
            'route_id': request.route_id,
            'user_id': request.user_id,
            'pois': [self._serialize_poi(poi) for poi in request.pois],
            'preferences': self._serialize_preferences(request.preferences) if request.preferences else {},
            'constraints': self._serialize_constraints(request.constraints) if request.constraints else {}
        }
    
    def _serialize_poi(self, poi):
        """Serializar POI para procesamiento"""
        return {
            'id': poi.id,
            'name': poi.name,
            'latitude': poi.latitude,
            'longitude': poi.longitude,
            'category': poi.category,
            'subcategory': poi.subcategory,
            'visit_duration': poi.visit_duration,
            'cost': poi.cost,
            'rating': poi.rating,
            'opening_hours': getattr(poi, 'opening_hours', None),
            'description': getattr(poi, 'description', None),
            'image_url': getattr(poi, 'image_url', None),
            'accessibility': poi.accessibility,
            'provider_id': poi.provider_id,
            'provider_name': poi.provider_name
        }
    
    def _serialize_preferences(self, preferences):
        """Serializar preferencias para procesamiento"""
        return {
            'optimize_for': preferences.optimize_for,
            'max_total_time': preferences.max_total_time,
            'max_total_cost': preferences.max_total_cost,
            'preferred_categories': list(preferences.preferred_categories),
            'avoid_categories': list(preferences.avoid_categories),
            'accessibility_required': preferences.accessibility_required
        }
    
    def _serialize_constraints(self, constraints):
        """Serializar restricciones para procesamiento"""
        return {
            'start_location': {
                'latitude': constraints.start_location.latitude,
                'longitude': constraints.start_location.longitude
            } if constraints.start_location else None,
            'end_location': {
                'latitude': constraints.end_location.latitude,
                'longitude': constraints.end_location.longitude
            } if constraints.end_location else None,
            'start_time': constraints.start_time,
            'lunch_break_required': constraints.lunch_break_required,
            'lunch_break_duration': constraints.lunch_break_duration
        }
    
    def _convert_results_to_grpc_response(self, route_id: str, results: Dict[str, Any], 
                                        execution_time: float, job_id: str) -> route_optimization_pb2.RouteOptimizationResponse:
        """Convertir resultados a respuesta gRPC"""
        response = route_optimization_pb2.RouteOptimizationResponse()
        
        # Información básica
        response.route_id = route_id
        response.job_id = job_id
        response.status = "COMPLETED"
        response.message = "Optimización completada exitosamente"
        
        # Resultados de optimización
        response.results.optimized_route_id = results.get('optimized_route_id', route_id)
        response.results.total_distance_km = results.get('total_distance_km', 0.0)
        response.results.total_time_minutes = results.get('total_time_minutes', 0)
        response.results.total_cost = results.get('total_cost', 0.0)
        response.results.optimization_score = results.get('optimization_score', 0.0)
        
        # Secuencia optimizada de POIs
        for poi_data in results.get('optimized_sequence', []):
            optimized_poi = response.results.optimized_sequence.add()
            optimized_poi.poi_id = poi_data.get('poi_id', 0)
            optimized_poi.name = poi_data.get('name', '')
            optimized_poi.latitude = poi_data.get('latitude', 0.0)
            optimized_poi.longitude = poi_data.get('longitude', 0.0)
            optimized_poi.visit_order = poi_data.get('visit_order', 0)
            optimized_poi.estimated_visit_time = poi_data.get('estimated_visit_time', 0)
            optimized_poi.arrival_time = poi_data.get('arrival_time', '')
            optimized_poi.departure_time = poi_data.get('departure_time', '')
        
        # Métricas de optimización
        response.results.metrics.execution_time_seconds = execution_time
        response.results.metrics.algorithm_used = results.get('optimization_algorithm', 'MRL-AMIS')
        response.results.metrics.iterations_completed = results.get('iterations_completed', 0)
        response.results.metrics.feasible_solutions_found = results.get('feasible_solutions_found', 0)
        response.results.metrics.generated_at = results.get('generated_at', '')
        
        return response

def serve():
    """Iniciar el servidor gRPC con sistema de colas"""
    port = os.getenv('GRPC_PORT', '50051')
    max_workers = int(os.getenv('GRPC_MAX_WORKERS', '10'))
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    route_optimization_pb2_grpc.add_RouteOptimizationServiceServicer_to_server(
        RouteOptimizationServicer(), server
    )
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Iniciando servidor gRPC en {listen_addr} con {max_workers} workers")
    server.start()
    logger.info("Servidor gRPC con sistema de colas iniciado correctamente")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Cerrando servidor gRPC...")
        
        # Cerrar sistema de colas
        from job_queue_system import shutdown_job_queue_system
        shutdown_job_queue_system()
        
        # Cerrar servidor gRPC
        server.stop(grace=5)
        logger.info("Servidor gRPC cerrado")

if __name__ == '__main__':
    serve()