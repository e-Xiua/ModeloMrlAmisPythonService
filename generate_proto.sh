#!/bin/bash

# Script para generar el código Python desde el archivo .proto

# Directorio donde se encuentra el archivo .proto
PROTO_DIR="proto"

# Directorio de salida para los archivos generados
OUTPUT_DIR="src/grpc_generated"

# Crear el directorio de salida si no existe
mkdir -p $OUTPUT_DIR

# Generar el código Python
python -m grpc_tools.protoc \
    -I$PROTO_DIR \
    --python_out=$OUTPUT_DIR \
    --grpc_python_out=$OUTPUT_DIR \
    $PROTO_DIR/route_optimization.proto

echo "Código gRPC generado exitosamente en $OUTPUT_DIR"

# Crear __init__.py para que sea un paquete Python
touch $OUTPUT_DIR/__init__.py

echo "✅ Generación completada"
