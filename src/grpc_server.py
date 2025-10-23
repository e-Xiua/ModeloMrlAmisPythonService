"""
Servidor gRPC para el Servicio de Optimización de Rutas MRL-AMIS
Implementa el servicio RouteOptimizationService definido en route_optimization.proto
"""

import grpc
from concurrent import futures
import time
import logging
import os
import sys

# Agregar el directorio src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar los módulos generados por protobuf
from grpc_generated import route_optimization_pb2
from grpc_generated import route_optimization_pb2_grpc

# Importar los módulos del algoritmo MRL-AMIS
# TODO: Ajustar estos imports según la estructura real de tu proyecto
# from ejecucion_iteraciones_mrl_amis.ejecutar_mrl_amis import ejecutar_mrl_amis
# from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import bucle_mrl_amis

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RouteOptimizationServicer(route_optimization_pb2_grpc.RouteOptimizationServiceServicer):
    """
    Implementación del servicio de optimización de rutas
    """

    def __init__(self):
        logger.info("Inicializando RouteOptimizationServicer...")
        self.version = "1.0.0"
        # Aquí puedes inicializar los componentes del algoritmo MRL-AMIS
        
    def OptimizeRoute(self, request, context):
        """
        Método principal que ejecuta la optimización de ruta usando MRL-AMIS
        
        Args:
            request: RouteOptimizationRequest con los POIs y preferencias
            context: Contexto de gRPC
            
        Returns:
            RouteOptimizationResponse con la ruta optimizada
        """
        logger.info(f"Recibida solicitud de optimización para ruta: {request.route_id}")
        logger.info(f"Usuario: {request.user_id}")
        logger.info(f"Número de POIs: {len(request.pois)}")
        
        start_time = time.time()
        
        try:
            # Extraer información del request
            pois = []
            for poi in request.pois:
                pois.append({
                    'id': poi.id,
                    'name': poi.name,
                    'latitude': poi.latitude,
                    'longitude': poi.longitude,
                    'category': poi.category,
                    'visit_duration': poi.visit_duration,
                    'cost': poi.cost,
                    'rating': poi.rating
                })
            
            logger.info(f"POIs procesados: {[p['name'] for p in pois]}")
            
            # TODO: Aquí va la integración real con el algoritmo MRL-AMIS
            # Por ahora, se genera una respuesta de ejemplo
            
            # Simulación de procesamiento
            logger.info("Ejecutando algoritmo MRL-AMIS...")
            time.sleep(2)  # Simular procesamiento
            
            # Crear respuesta de ejemplo
            response = route_optimization_pb2.RouteOptimizationResponse(
                route_id=request.route_id,
                success=True,
                message="Optimización completada exitosamente"
            )
            
            # Crear secuencia optimizada (ejemplo)
            for idx, poi in enumerate(request.pois[:5]):  # Limitar a 5 POIs para el ejemplo
                optimized_poi = route_optimization_pb2.OptimizedPOI(
                    poi_id=poi.id,
                    poi_name=poi.name,
                    visit_order=idx + 1,
                    arrival_time=f"{9 + idx}:00",
                    departure_time=f"{9 + idx}:{poi.visit_duration}",
                    estimated_visit_time=poi.visit_duration,
                    latitude=poi.latitude,
                    longitude=poi.longitude
                )
                response.results.optimized_sequence.append(optimized_poi)
            
            # Agregar resultados generales
            response.results.total_distance_km = 45.5
            response.results.total_time_minutes = 240
            response.results.total_cost = 150.0
            response.results.optimization_score = 0.87
            response.results.route_description = "Ruta optimizada usando algoritmo MRL-AMIS"
            
            # Agregar métricas de optimización
            execution_time = time.time() - start_time
            response.metrics.hypervolume = 0.75
            response.metrics.arp = 0.82
            response.metrics.spacing = 0.15
            response.metrics.pareto_front_size = 10
            response.metrics.total_iterations = 50
            response.metrics.execution_time_seconds = execution_time
            
            logger.info(f"Optimización completada en {execution_time:.2f} segundos")
            logger.info(f"Ruta optimizada con {len(response.results.optimized_sequence)} POIs")
            
            return response
            
        except Exception as e:
            logger.error(f"Error durante la optimización: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error interno del servidor: {str(e)}")
            
            # Retornar respuesta de error
            return route_optimization_pb2.RouteOptimizationResponse(
                route_id=request.route_id,
                success=False,
                message=f"Error durante la optimización: {str(e)}"
            )

    def HealthCheck(self, request, context):
        """
        Verifica el estado del servicio
        
        Args:
            request: HealthRequest
            context: Contexto de gRPC
            
        Returns:
            HealthResponse con el estado del servicio
        """
        logger.info(f"Health check solicitado por: {request.service_name}")
        
        return route_optimization_pb2.HealthResponse(
            is_healthy=True,
            status="OK",
            version=self.version
        )


def serve():
    """
    Inicia el servidor gRPC
    """
    port = os.getenv('GRPC_PORT', '50051')
    
    # Crear servidor gRPC con pool de threads
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Agregar el servicer al servidor
    route_optimization_pb2_grpc.add_RouteOptimizationServiceServicer_to_server(
        RouteOptimizationServicer(), server
    )
    
    # Configurar el puerto
    server.add_insecure_port(f'[::]:{port}')
    
    # Iniciar el servidor
    server.start()
    logger.info(f"🚀 Servidor gRPC iniciado en puerto {port}")
    logger.info(f"📡 Esperando solicitudes de optimización de rutas...")
    
    try:
        # Mantener el servidor en ejecución
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Deteniendo servidor gRPC...")
        server.stop(0)
        logger.info("Servidor detenido")


if __name__ == '__main__':
    serve()
