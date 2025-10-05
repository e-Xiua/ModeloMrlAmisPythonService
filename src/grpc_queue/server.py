"""Servidor gRPC con sistema de colas usando dataclasses de dominio."""

import grpc
from concurrent import futures
import time
import logging
import sys
import os
from typing import Dict, Any, List

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Agregar el directorio 'src' (padre de este paquete) al sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # -> .../ModeloMrlAmisPythonService/src
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
from generated import route_optimization_pb2 
from generated import route_optimization_pb2_grpc 

from grpc_queue.queue_system import get_queue_system, shutdown_queue_system

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RouteOptimizationServicer(route_optimization_pb2_grpc.RouteOptimizationServiceServicer):
    
    def __init__(self):
        """Inicializar el servicio de optimización de rutas con sistema de colas y dataclasses"""
        logger.info("Inicializando RouteOptimizationService con dataclasses...")
        
        # Sistema de colas actualizado
        self.job_queue = get_queue_system()
        
        # Configuración predeterminada del modelo MRL-AMIS
        self.default_config = {
            'num_work_packages': 100,
            'max_iterations': 100,
            'num_pois': 15,
            'max_pois_per_route': 10,
            'min_pois_per_route': 1,
            'num_routes_per_wp': 3,
            'max_duration_per_route': 72000,  # 20 horas
            'maximize_objectives_list': [True, False, False, True, False],
            'ref_point_hypervolume': [0, 1000, 1000, -1000, 1000]
        }
        
        logger.info("RouteOptimizationService inicializado correctamente con dataclasses")
    
    def OptimizeRoute(self, request, context):
        """Método principal para optimizar rutas usando MRL-AMIS con dataclasses"""
        try:
            
            # LOG 1: Solicitud gRPC entrante
            logger.info("======================================================================")
            logger.info(f"PASO 1: Solicitud gRPC recibida (ID: {request.route_id})")
            logger.info(f"  -> Contenido gRPC (crudo): {request}")
            
            logger.info(f"=== NUEVA SOLICITUD DE OPTIMIZACIÓN ===")
            logger.info(f"Route ID: {request.route_id}")
            logger.info(f"User ID: {request.user_id}")
            logger.info(f"Número de POIs: {len(request.pois)}")
            
            # Validar solicitud
            if not request.pois:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Al menos un POI es requerido')
                return route_optimization_pb2.RouteOptimizationResponse()
            
            # Preparar datos del trabajo
            job_data = {
                'route_id': request.route_id or f'route_{int(time.time())}',
                'user_id': request.user_id or 'anonymous'
            }
            
            # Serializar datos de la solicitud con manejo seguro de campos opcionales
            request_data = self._serialize_grpc_request(request)
            
            # LOG 2: Datos serializados a diccionario
            logger.info(f"PASO 2: Solicitud serializada a diccionario (ID: {request.route_id})")
            logger.info(f"  -> Contenido serializado: {request_data}")
            
            # Normalizar restricciones con POIs si es necesario
            request_data = self._ensure_default_locations(request_data)
            
            # LOG 3: Datos normalizados antes de encolar
            logger.info(f"PASO 3: Diccionario normalizado antes de encolar (ID: {request.route_id})")
            logger.info(f"  -> Contenido normalizado: {request_data}")
            
            # Enviar trabajo a la cola
            job_id = self.job_queue.submit_job(job_data, request_data)
            
            logger.info(f"PASO 4: Trabajo {job_id} enviado a la cola.")
            logger.info("======================================================================")
            
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
            response.estimated_wait_time_minutes = queue_info['queue_size'] * 5
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando solicitud de optimización: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno del servidor: {str(e)}')
            return route_optimization_pb2.RouteOptimizationResponse()
        
    def _ensure_default_locations(self, request_data: dict) -> dict:
        """Asegura que existan ubicaciones por defecto si no se proporcionan."""
        constraints = request_data.get('constraints') or {}
        pois = request_data.get('pois') or []
        
        if not constraints.get('start_location') and pois:
            first_poi = pois[0]
            constraints['start_location'] = {
                'latitude': first_poi.get('latitude', 0.0),
                'longitude': first_poi.get('longitude', 0.0)
            }
        
        if not constraints.get('end_location') and len(pois) > 1:
            last_poi = pois[-1]
            constraints['end_location'] = {
                'latitude': last_poi.get('latitude', 0.0),
                'longitude': last_poi.get('longitude', 0.0)
            }
        
        request_data['constraints'] = constraints
        return request_data

    def _get_default_start_location(self, constraints, pois):
        if constraints.get('start_location'):
            return constraints.get('start_location')
        if pois:
            start_loc = self._poi_to_loc(pois[0])
            if start_loc and (start_loc['latitude'] or start_loc['longitude']):
                return start_loc
        return None

    def _get_default_end_location(self, constraints, pois):
        if constraints.get('end_location'):
            return constraints.get('end_location')
        if len(pois) > 1:
            end_loc = self._poi_to_loc(pois[-1])
            if end_loc and (end_loc['latitude'] or end_loc['longitude']):
                return end_loc
        return None
    
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
                response.message = f"No se pudo cancelar el trabajo {job_id}"
            
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
                status=f"Service running with dataclasses - Queue: {queue_info['queue_size']} waiting, {queue_info['active_jobs']} active",
                version="2.1.0",
                queue_size=queue_info['queue_size'],
                active_jobs=queue_info['active_jobs']
            )
        except Exception as e:
            logger.error(f"Error en health check: {str(e)}")
            return route_optimization_pb2.HealthResponse(
                is_healthy=False,
                status=f"Service error: {str(e)}",
                version="2.1.0"
            )
    
    def _serialize_grpc_request(self, request) -> Dict[str, Any]:
        """Serializar solicitud gRPC para procesamiento en cola con manejo seguro de campos"""
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
        """Serializar preferencias para procesamiento con manejo seguro de preferred_categories"""
        return {
            'optimize_for': preferences.optimize_for,
            'max_total_time': preferences.max_total_time,
            'max_total_cost': preferences.max_total_cost,
            'preferred_categories': list(getattr(preferences, 'preferred_categories', [])),
            'avoid_categories': list(getattr(preferences, 'avoid_categories', [])),
            'accessibility_required': preferences.accessibility_required
        }
    
    def _serialize_constraints(self, constraints):
        """Serializar restricciones para procesamiento de forma segura."""
        start_loc = None
        end_loc = None
# FORMA SEGURA de comprobar si el campo existe antes de acceder a él
        if constraints and constraints.HasField("start_location"):
            logger.info("  -> Serializando `start_location` desde la solicitud gRPC.")
            start_loc = {
                'latitude': constraints.start_location.latitude,
                'longitude': constraints.start_location.longitude
            }
        else:
            logger.info("  -> No se encontró `start_location` en la solicitud gRPC. Se usará `None`.")

        if constraints and constraints.HasField("end_location"):
            logger.info("  -> Serializando `end_location` desde la solicitud gRPC.")
            end_loc = {
                'latitude': constraints.end_location.latitude,
                'longitude': constraints.end_location.longitude
            }
        else:
            logger.info("  -> No se encontró `end_location` en la solicitud gRPC. Se usará `None`.")
        return {
            'start_location': start_loc,
            'end_location': end_loc,
            'start_time': constraints.start_time if constraints else '',
            'lunch_break_required': constraints.lunch_break_required if constraints else False,
            'lunch_break_duration': constraints.lunch_break_duration if constraints else 0
        }
    
    def _convert_results_to_grpc_response(self, route_id: str, results: Dict[str, Any], 
                                        execution_time: float, job_id: str) -> route_optimization_pb2.RouteOptimizationResponse:
        """Convertir resultados a respuesta gRPC"""
        response = route_optimization_pb2.RouteOptimizationResponse()
        
        # Información básica
        response.route_id = route_id
        response.job_id = job_id
        response.status = "COMPLETED"
        response.message = "Optimización completada exitosamente usando dataclasses"
        
        # Resultados de optimización
        response.results.total_distance_km = results.get('total_distance_km', 0.0)
        response.results.total_time_minutes = results.get('total_time_minutes', 0)
        response.results.total_cost = results.get('total_cost', 0.0)
        response.results.optimization_score = results.get('optimization_score', 0.0)
        
        # Secuencia optimizada de POIs
        for poi_data in results.get('optimized_sequence', []):
            optimized_poi = response.results.optimized_sequence.add()
            optimized_poi.poi_id = poi_data.get('poi_id', 0)
            optimized_poi.poi_name = poi_data.get('name', '')
            optimized_poi.latitude = poi_data.get('latitude', 0.0)
            optimized_poi.longitude = poi_data.get('longitude', 0.0)
            optimized_poi.visit_order = poi_data.get('visit_order', 0)
            optimized_poi.estimated_visit_time = poi_data.get('estimated_visit_time', 0)
            optimized_poi.arrival_time = poi_data.get('arrival_time', '')
            optimized_poi.departure_time = poi_data.get('departure_time', '')
        
        # Métricas de optimización
            try:
                response.metrics.hypervolume = float(results.get('hypervolume', 0.0))
                response.metrics.arp = float(results.get('arp', 0.0))              
                response.metrics.spacing = float(results.get('spacing', 0.0))
                response.metrics.pareto_front_size = int(results.get('pareto_front_size', 0))
                response.metrics.total_iterations = int(results.get('total_iterations', 0))
                response.metrics.execution_time_seconds = float(execution_time)
                
                # Campos adicionales si existen
                if 'iterations_completed' in results:
                    response.metrics.total_iterations = int(results.get('iterations_completed', 0))
                if 'feasible_solutions_found' in results:
                    response.metrics.pareto_front_size = int(results.get('feasible_solutions_found', 0))
            except AttributeError as e:
                # Si el campo metrics no existe en el proto, simplemente continuar
                logger.warning(f"Campo 'metrics' no disponible en RouteOptimizationResponse: {e}")
        
        return response


def serve():
    """Iniciar el servidor gRPC con sistema de colas y dataclasses"""
    port = os.getenv('GRPC_PORT', '50051')
    max_workers = int(os.getenv('GRPC_MAX_WORKERS', '10'))
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    route_optimization_pb2_grpc.add_RouteOptimizationServiceServicer_to_server(
        RouteOptimizationServicer(), server
    )
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Iniciando servidor gRPC con dataclasses en {listen_addr} con {max_workers} workers")
    server.start()
    logger.info("Servidor gRPC con sistema de colas y dataclasses iniciado correctamente")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Cerrando servidor gRPC...")
        
        # Cerrar sistema de colas
        shutdown_queue_system()
        
        # Cerrar servidor gRPC
        server.stop(grace=5)
        logger.info("Servidor gRPC cerrado")


if __name__ == '__main__':
    serve()