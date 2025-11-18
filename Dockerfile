# Usar imagen base de Python
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    build-essential \
    libboost-all-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements.txt .
COPY setup.py .
COPY setup.cfg .
COPY pyproject.toml .

# Copiar código fuente y proto files
COPY src ./src
COPY data ./data
COPY proto ./proto

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openpyxl scikit-learn matplotlib seaborn folium polyline && \
    pip install --no-cache-dir -e .

# Generar código Python desde proto files
RUN mkdir -p src/grpc_generated && \
    python -m grpc_tools.protoc \
    -Iproto \
    --python_out=src/grpc_generated \
    --grpc_python_out=src/grpc_generated \
    proto/route_optimization.proto && \
    touch src/grpc_generated/__init__.py

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 mrluser && \
    chown -R mrluser:mrluser /app

USER mrluser

# Exponer puerto gRPC
EXPOSE 50051

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV GRPC_PORT=50051

# Health check - Verify gRPC server is listening on port 50051
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5); result = s.connect_ex(('localhost', 50051)); s.close(); exit(0 if result == 0 else 1)" || exit 1

# Comando por defecto - Ejecutar servidor gRPC con módulo grpc_queue
CMD ["python", "-m", "grpc_queue.server"]
