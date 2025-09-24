#!/usr/bin/env python3
"""
Script de prueba rápida para verificar el sistema OOP
"""

import sys
sys.path.append('src')

def test_basic_functionality():
    """Prueba básica del sistema OOP"""
    
    print("🧪 Probando sistema OOP para MRL-AMIS...")
    
    try:
        # Importar módulos
        from runmodel.models import POI, TouristGroup, generate_synthetic_data
        from runmodel.util import generate_data
        
        print("✓ Imports exitosos")
        
        # Generar datos
        data = generate_data(num_pois=10, seed=42)
        print("✓ Datos generados exitosamente")
        
        # Verificar estructura
        expected_keys = [
            'pois', 'tourist_groups', 'distances', 'travel_times', 'costs',
            'co2_emission_cost', 'accident_risk', 'preferencias_grupos_turistas'
        ]
        
        for key in expected_keys:
            if key in data:
                print(f"✓ {key}: {type(data[key])}")
            else:
                print(f"✗ Falta: {key}")
        
        # Verificar dimensiones
        n_pois = len(data['pois'])
        n_groups = len(data['tourist_groups'])
        
        print(f"✓ POIs: {n_pois}")
        print(f"✓ Grupos: {n_groups}")
        print(f"✓ Matriz distancias: {data['distances'].shape}")
        print(f"✓ Matriz preferencias: {data['preferencias_grupos_turistas'].shape}")
        
        # Verificar que no hay valores nulos críticos
        critical_matrices = ['distances', 'travel_times', 'costs']
        for matrix_name in critical_matrices:
            if matrix_name in data:
                null_count = data[matrix_name].isnull().sum().sum()
                print(f"✓ {matrix_name}: {null_count} valores nulos")
        
        print("\n🎉 Sistema OOP funcionando correctamente!")
        print(f"📊 Listo para usar con {n_pois} POIs y {n_groups} grupos turísticos")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
