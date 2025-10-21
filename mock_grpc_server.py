import grpc
from concurrent import futures
import time
import logging
import sys
import os

# Configurar PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))

# Importar protobuf generados
try:
    import route_optimization_pb2 as pb2
    import route_optimization_pb2_grpc as pb2_grpc
except ImportError as e:
    print(f"Error importing protobuf files: {e}")
    print("Make sure to run: python -m grpc_tools.protoc -I./protos --python_out=./generated --grpc_python_out=./generated ./protos/route_optimization.proto")
    sys.exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockRouteOptimizationServicer(pb2_grpc.RouteOptimizationServiceServicer):
    """Servidor gRPC Mock para probar la comunicación sin dependencias del modelo MRL-AMIS"""
    
    def OptimizeRoute(self, request, context):
        """Procesar solicitud de optimización con datos simulados"""
        logger.info("=== MOCK OPTIMIZATION REQUEST RECEIVED ===")
        logger.info(f"Route ID: {request.route_id}")
        logger.info(f"User ID: {request.user_id}")
        logger.info(f"Number of POIs: {len(request.pois)}")
        
        # Log POIs received
        for i, poi in enumerate(request.pois):
            logger.info(f"POI {i+1}: {poi.name} at ({poi.latitude}, {poi.longitude})")
        
        # Simular procesamiento (sin modelo real)
        time.sleep(2)  # Simular tiempo de procesamiento
        
        # Crear respuesta mock con datos optimizados simulados
        optimized_sequence = []
        
        # Crear secuencia optimizada simple (revertir orden como ejemplo)
        for i, poi in enumerate(reversed(request.pois)):
            optimized_poi = pb2.OptimizedPOI(
                poi_id=poi.id,
                poi_name=poi.name,
                visit_order=i + 1,
                arrival_time=f"{8 + i * 2}:00",  # Hora simulada
                departure_time=f"{8 + i * 2 + 1}:30",  # Salida simulada
                estimated_visit_time=90,  # 1.5 horas
                latitude=poi.latitude,
                longitude=poi.longitude
            )
            optimized_sequence.append(optimized_poi)
        
        # Crear resultados simulados
        total_distance = len(request.pois) * 15.5  # Distancia simulada
        total_time = len(request.pois) * 120  # Tiempo simulado
        optimization_score = 0.85  # Score simulado
        
        results = pb2.OptimizationResults(
            optimized_sequence=optimized_sequence,
            total_distance_km=total_distance,
            total_time_minutes=total_time,
            total_cost=sum(poi.cost for poi in request.pois),
            optimization_score=optimization_score,
            route_description=f"Ruta mock optimizada con {len(request.pois)} POIs"
        )
        
        # Crear métricas simuladas
        metrics = pb2.OptimizationMetrics(
            hypervolume=0.95,
            arp=0.88,
            spacing=0.12,
            pareto_front_size=10,
            total_iterations=50,
            execution_time_seconds=2.0
        )
        
        response = pb2.RouteOptimizationResponse(
            route_id=request.route_id,
            success=True,
            message=f"Mock optimization completed successfully for {len(request.pois)} POIs",
            results=results,
            metrics=metrics
        )
        
        logger.info("=== MOCK OPTIMIZATION RESPONSE SENT ===")
        logger.info(f"Optimized {len(optimized_sequence)} POIs")
        logger.info(f"Total distance: {total_distance} km")
        logger.info(f"Total time: {total_time} minutes")
        
        return response
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        logger.info(f"Health check requested by {request.service_name}")
        return pb2.HealthResponse(
            is_healthy=True,
            status="MockRouteOptimizationService is running",
            version="1.0.0-mock"
        )

def serve():
    """Start gRPC server"""
    port = os.getenv('GRPC_PORT', '50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    pb2_grpc.add_RouteOptimizationServiceServicer_to_server(
        MockRouteOptimizationServicer(), server
    )
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"🚀 Starting Mock gRPC server on {listen_addr}")
    server.start()
    logger.info("✅ Mock gRPC server started successfully")
    logger.info("📝 Ready to receive optimization requests...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("🛑 Stopping Mock gRPC server...")
        server.stop(grace=5)

if __name__ == '__main__':
    serve()