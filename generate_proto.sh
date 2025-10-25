<<<<<<< HEAD
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
=======
#!/usr/bin/env bash
# filepath: /home/santiagovera/FrontEnd/e-Xiua/ModeloMrlAmisPythonService/generate_proto.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/src/protos"
OUT_DIR="${SCRIPT_DIR}/src/generated"

echo "Generando archivos protobuf para gRPC..."
echo "PROTO_DIR=${PROTO_DIR}"
echo "OUT_DIR=${OUT_DIR}"

if [[ ! -f "${PROTO_DIR}/route_optimization.proto" ]]; then
  echo "❌ No se encontró ${PROTO_DIR}/route_optimization.proto"
  exit 1
fi

mkdir -p "${OUT_DIR}"
# Asegura paquetes Python
touch "${SCRIPT_DIR}/src/__init__.py"
touch "${OUT_DIR}/__init__.py"

# Usa grpc_tools.protoc desde el entorno conda activo
python -m grpc_tools.protoc \
  -I "${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/route_optimization.proto"

# Parche: usar import relativo en el stub gRPC
sed -i 's/^import route_optimization_pb2 as /from \. import route_optimization_pb2 as /' "${OUT_DIR}/route_optimization_pb2_grpc.py"

echo "✅ Protobuf generado en ${OUT_DIR}"
>>>>>>> origin/primera-version-optimizacion
