#!/bin/bash

# Script para ejecutar el servidor gRPC con el nuevo paquete grpc_queue

echo "=== Iniciando servidor gRPC con dataclasses ==="
echo "Puerto: ${GRPC_PORT:-50051}"
echo "Workers: ${GRPC_MAX_WORKERS:-10}"
echo ""

# Cambiar al directorio del proyecto
cd "$(dirname "$0")"

# Verificar que los protobuf están generados
if [ ! -f "generated/route_optimization_pb2.py" ]; then
    echo "Generando archivos protobuf..."
    ./generate_proto.sh
fi

# Ejecutar servidor usando el nuevo paquete
echo "Ejecutando servidor gRPC con dataclasses..."
python -m grpc_queue.server