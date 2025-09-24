#!/usr/bin/env python3
"""
Script de ejemplo que demuestra el uso del sistema OOP para generar datos
en lugar de cargar desde archivos Excel.
"""

from runmodel.util import generate_data, cargar_datos
from runmodel.models import POI, TouristGroup, create_sample_pois, create_sample_tourist_groups
import pandas as pd
import numpy as np

def ejemplo_uso_oop():
    """Ejemplo de uso del sistema OOP para datos sintéticos"""
    
    print("="*60)
    print("EJEMPLO DE USO DEL SISTEMA OOP PARA DATOS MRL-AMIS")
    print("="*60)
    
    # Método 1: Usar generate_data() - Reemplazo directo de cargar_datos()
    print("\n1. Generando datos sintéticos (reemplazo de cargar_datos()):")
    print("-" * 50)
    
    data = generate_data(num_pois=15, seed=42)
    
    print(f"\nDatos generados:")
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key}: DataFrame {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: Dict con {len(value)} elementos")
        else:
            print(f"  {key}: {type(value)}")
    
    # Método 2: Usar objetos POI directamente
    print("\n\n2. Creando POIs usando objetos OOP:")
    print("-" * 40)
    
    # Crear POIs de ejemplo
    pois = create_sample_pois()[:5]  # Solo los primeros 5
    
    for poi in pois:
        print(f"POI: {poi.name}")
        print(f"  Posición: {poi.position}")
        print(f"  Tipo: {poi.tipo_poi}")
        print(f"  Preferencia base: {poi.preferencia:.1f}")
        print(f"  Sostenibilidad combinada: {poi.combined_sustainability:.1f}")
        print(f"  Riesgo total: {poi.total_risk:.1f}")
        print()
    
    # Método 3: Crear grupos turísticos
    print("\n3. Creando grupos turísticos usando objetos OOP:")
    print("-" * 45)
    
    grupos = create_sample_tourist_groups()[:3]  # Solo los primeros 3
    
    for grupo in grupos:
        print(f"Grupo: {grupo.grupo_id}")
        print(f"  Tipo: {grupo.tipo_turista}")
        print(f"  Tamaño: {grupo.tamano_grupo} personas")
        print(f"  Presupuesto: ${grupo.presupuesto_min_usd:.0f} - ${grupo.presupuesto_max_usd:.0f}")
        print(f"  Tiempo disponible: {grupo.tiempo_disponible_min} minutos")
        print(f"  Preferencias tipos: {grupo.preferencias_tipos}")
        print()
    
    # Método 4: Calcular preferencias específicas
    print("\n4. Calculando preferencias específicas grupo-POI:")
    print("-" * 50)
    
    grupo_aventurero = grupos[0]  # grupo_1 es aventurero
    grupo_cultural = grupos[1]    # grupo_2 es cultural
    
    poi_aventura = next(poi for poi in pois if poi.tipo_poi == 'aventura')
    poi_cultura = next(poi for poi in pois if poi.tipo_poi == 'cultura')
    
    print(f"POI Aventura: {poi_aventura.name}")
    print(f"  Preferencia grupo aventurero: {grupo_aventurero.calculate_poi_preference(poi_aventura.tipo_poi, poi_aventura.preferencia):.1f}")
    print(f"  Preferencia grupo cultural: {grupo_cultural.calculate_poi_preference(poi_aventura.tipo_poi, poi_aventura.preferencia):.1f}")
    
    print(f"\nPOI Cultura: {poi_cultura.name}")
    print(f"  Preferencia grupo aventurero: {grupo_aventurero.calculate_poi_preference(poi_cultura.tipo_poi, poi_cultura.preferencia):.1f}")
    print(f"  Preferencia grupo cultural: {grupo_cultural.calculate_poi_preference(poi_cultura.tipo_poi, poi_cultura.preferencia):.1f}")
    
    # Método 5: Verificar matrices generadas
    print("\n\n5. Verificando matrices generadas:")
    print("-" * 35)
    
    print("Matriz de distancias (primeros 5x5 POIs, en km):")
    distances = data['distances'].iloc[:5, :5]
    print(distances.round(1))
    
    print("\nMatriz de tiempos de viaje (primeros 5x5 POIs, en minutos):")
    times = data['travel_times'].iloc[:5, :5]
    print(times.round(1))
    
    print("\nMatriz de costos (primeros 5x5 POIs, en USD):")
    costs = data['costs'].iloc[:5, :5]
    print(costs.round(2))
    
    print("\nMatriz de preferencias de grupos por POIs (primeros 3 grupos, 5 POIs):")
    preferences = data['preferencias_grupos_turistas'].iloc[:3, :5]
    print(preferences.round(1))
    
    print("\n" + "="*60)
    print("DATOS LISTOS PARA USAR EN MRL-AMIS")
    print("="*60)
    print("\nPuedes usar estos datos directamente en tu algoritmo:")
    print("  data = generate_data(num_pois=15)")
    print("  # data tiene la misma estructura que cargar_datos()")
    print("  # pero sin necesidad de archivos Excel")
    
    return data

def comparar_con_excel():
    """Compara la estructura de datos generados vs cargados desde Excel"""
    
    print("\n" + "="*60)
    print("COMPARACIÓN: DATOS SINTÉTICOS VS EXCEL")
    print("="*60)
    
    try:
        # Intentar cargar datos desde Excel
        print("\nIntentando cargar desde Excel...")
        data_excel = cargar_datos()
        if data_excel:
            print("✓ Datos cargados desde Excel exitosamente")
            excel_keys = set(data_excel.keys())
        else:
            print("✗ No se pudieron cargar datos desde Excel")
            excel_keys = set()
    except Exception as e:
        print(f"✗ Error cargando desde Excel: {e}")
        excel_keys = set()
    
    # Generar datos sintéticos
    print("Generando datos sintéticos...")
    data_synthetic = generate_data(num_pois=15)
    print("✓ Datos sintéticos generados exitosamente")
    synthetic_keys = set(data_synthetic.keys())
    
    print(f"\nClaves en datos sintéticos: {sorted(synthetic_keys)}")
    if excel_keys:
        print(f"Claves en datos Excel: {sorted(excel_keys)}")
        
        common_keys = excel_keys & synthetic_keys
        only_excel = excel_keys - synthetic_keys
        only_synthetic = synthetic_keys - excel_keys
        
        print(f"\nClaves comunes: {sorted(common_keys)}")
        if only_excel:
            print(f"Solo en Excel: {sorted(only_excel)}")
        if only_synthetic:
            print(f"Solo en sintéticos: {sorted(only_synthetic)}")
    
    return data_synthetic

if __name__ == "__main__":
    # Ejecutar ejemplo principal
    data = ejemplo_uso_oop()
    
    # Comparar con Excel si está disponible
    data_comparison = comparar_con_excel()
    
    print(f"\n🎉 Sistema OOP listo para usar!")
    print(f"📊 {len(data['pois'])} POIs y {len(data['tourist_groups'])} grupos generados")
    print(f"🔄 Reemplaza cargar_datos() con generate_data()")
