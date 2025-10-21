#!/bin/bash

echo "🐍 Iniciando servidor gRPC Python MRL-AMIS..."

# Cambiar al directorio del proyecto Python
cd /home/santiagovera/FrontEnd/e-Xiua/ModeloMrlAmisPythonService

# Verificar que existe el archivo requirements
if [ ! -f "requirements_grpc.txt" ]; then
    echo "❌ Archivo requirements_grpc.txt no encontrado"
    exit 1
fi

# Instalar dependencias si es necesario
echo "📦 Instalando dependencias de gRPC..."
pip install -r requirements_grpc.txt

# Generar archivos protobuf Python si no existen
if [ ! -d "generated" ] || [ ! -f "generated/route_optimization_pb2.py" ]; then
    echo "🔧 Generando archivos protobuf Python..."
    ./generate_proto.sh
fi

# Verificar que los archivos protobuf fueron generados
if [ ! -f "generated/route_optimization_pb2.py" ] || [ ! -f "generated/route_optimization_pb2_grpc.py" ]; then
    echo "❌ Error: Archivos protobuf no generados correctamente"
    exit 1
fi

# Configurar PYTHONPATH para incluir el directorio generated
export PYTHONPATH="${PYTHONPATH}:$(pwd)/generated:$(pwd)/src"

echo "🚀 Iniciando servidor gRPC en puerto 50051..."
echo "📝 Logs del servidor:"
echo "=========================="

# Iniciar el servidor gRPC
python grpc_server.py