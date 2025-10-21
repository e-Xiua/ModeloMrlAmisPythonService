# Changelog: Reestructuración del Servidor gRPC con Sistema de Colas

## Resumen de Cambios

Esta reestructuración soluciona el error `preferred_categories` y reorganiza completamente el servidor gRPC en un paquete dedicado con mejor mapeo de datos.

## ✅ Cambios Completados

### 1. **Paquete `grpc_queue/` Creado**
- **Archivo**: `grpc_queue/__init__.py` - Inicialización del paquete
- **Archivo**: `grpc_queue/server.py` - Servidor gRPC refactorizado con manejo seguro de campos
- **Archivo**: `grpc_queue/queue_system.py` - Sistema de colas mejorado
- **Archivo**: `grpc_queue/domain_worker.py` - Worker MRL-AMIS usando dataclasses
- **Archivo**: `grpc_queue/data_mappers.py` - **✨ NUEVO** Mapeo dict→dataclasses completo

### 2. **Solucionado: Error `preferred_categories`**
```python
# ANTES (ERROR):
preferred_categories = preferences.preferred_categories  # AttributeError

# DESPUÉS (SOLUCIONADO):
preferred_categories = list(getattr(preferences, 'preferred_categories', []))
```

### 3. **Solucionado: Campo `start_location`**
```python
# ANTES - Ignoraba start_location:
origen = pois[0].id if pois else "1"

# DESPUÉS - Usa start_location de constraints:
start_location = constraints.get("start_location")
if start_location and start_location.get("latitude") and start_location.get("longitude"):
    origen = _find_closest_poi_to_location(pois, start_location) if pois else "1"
else:
    origen = pois[0].id if pois else "1"
```

### 4. **Nueva Matriz de CO2 Implementada**
```python
def _build_co2_matrix(pois: List[POI], travel_time_matrix: pd.DataFrame) -> pd.DataFrame:
    """Genera matriz de emisiones de CO2 basada en tiempos de viaje."""
    # CO2 por transporte: tiempo * factor emisión
    transport_co2 = travel_time_minutes * 0.02  # kg CO2 por minuto
    # CO2 por estadía: tiempo estadía * co2_per_hour del POI
    stay_co2 = stay_time_hours * target_poi.co2_per_hour
    co2_emissions[i, j] = transport_co2 + stay_co2
```

### 5. **DTOs Java Mejorados**
**Archivo actualizado**: `route-optimizer-service/src/main/java/com/exiua/routeoptimizer/dto/RouteProcessingRequestDTO.java`

Nuevos campos en `RoutePreferencesDTO`:
- `preferred_categories: List<String>`
- `avoid_categories: List<String>`
- `group_size: Integer`
- `tourist_type: String`
- `adventure_level: Double`
- `cost_sensitivity: Double`
- `sustainability_min: Double`
- `max_distance_km: Double`

Nuevos campos en `RouteConstraintsDTO`:
- `start_location: LocationDTO`
- `end_location: LocationDTO`

## 🔧 Arquitectura del Nuevo Sistema

### Flujo de Datos Mejorado
```
gRPC Request → data_mappers.py → POI/TouristGroup dataclasses → domain_worker.py → MRL-AMIS
     ↓
Serialización segura con getattr() → Sin más errores preferred_categories
     ↓  
start_location/end_location procesados correctamente → Origen/destino precisos
     ↓
Matrices generadas: distancia, tiempo, CO2 → Pipeline completo
```

### Nuevas Funciones Clave

1. **`build_domain_payload()`** - Construye toda la estructura para el worker
2. **`_map_poi_dict()`** - Transforma dict gRPC → dataclass POI
3. **`_map_tourist_group()`** - Transforma dict gRPC → dataclass TouristGroup
4. **`_find_closest_poi_to_location()`** - Encuentra POI más cercano a coordenadas
5. **`_build_co2_matrix()`** - Calcula emisiones CO2 por ruta

### Clase `DomainPayload` - Estructura de Datos
```python
@dataclass
class DomainPayload:
    pois: List[POI]                                    # Dataclasses de dominio
    tourist_group: TouristGroup                        # Grupo turístico mapeado
    pois_dataframe: pd.DataFrame                       # Para algoritmo MRL-AMIS
    distance_matrix: pd.DataFrame                      # Distancias Haversine
    travel_time_matrix: pd.DataFrame                   # Tiempos de viaje
    co2_matrix: pd.DataFrame                          # ✨ NUEVO: Emisiones CO2
    group_preferences_matrix: Dict[str, Dict[str, float]]  # Matriz preferencias
    internal_to_external_ids: Dict[str, str]          # Mapeo IDs internos↔externos
    raw_request: Dict[str, Any]                       # Request original para debug
```

## 🚀 Compatibilidad con Data Generator

El `data_generator.py` existente ya incluye:
- ✅ Matriz de CO2 (`calculate_co2_matrix()`)
- ✅ Datos mock consistentes con el nuevo mapeo
- ✅ Estructura compatible con dataclasses de dominio

## 📝 Próximos Pasos Recomendados

1. **Pruebas de integración** - Validar flujo completo gRPC → MRL-AMIS
2. **Optimización de rendimiento** - Cachear matrices para POIs frecuentes  
3. **Monitoreo** - Agregar métricas de tiempo de mapeo y procesamiento
4. **Documentación** - Actualizar README con nueva arquitectura

## 🐛 Errores Solucionados

- ❌ `AttributeError: preferred_categories` → ✅ Manejo seguro con `getattr()`
- ❌ `start_location` ignorado → ✅ Mapeo correcto a origen/destino
- ❌ Matriz CO2 faltante → ✅ Implementada con transporte + estadía
- ❌ DTOs incompletos → ✅ Campos completos para todas las preferencias

---

**Resumen**: Sistema completamente refactorizado, errores solucionados, y arquitectura mejorada para mayor mantenibilidad y funcionalidad.