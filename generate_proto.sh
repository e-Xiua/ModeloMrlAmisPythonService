#!/bin/bash

# Script para generar archivos protobuf para gRPC

echo "Generando archivos protobuf para gRPC..."

# Crear directorio para archivos generados si no existe
mkdir -p generated

# Generar archivos Python desde el archivo .proto
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ./protos/route_optimization.proto

if [ $? -eq 0 ]; then
    echo "✅ Archivos protobuf generados exitosamente en ./generated/"
    echo "📁 Archivos generados:"
    ls -la ./generated/
else
    echo "❌ Error generando archivos protobuf"
    exit 1
fi

echo "🚀 Para usar el servidor gRPC, ejecuta:"
echo "   python grpc_server.py"