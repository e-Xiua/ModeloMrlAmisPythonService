
from .models.data_generator import generate_synthetic_data
import pandas as pd
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import pickle
from AgenteQLearning.QLearningAgent import QLearningAgent
from analisis_multi_objetivo_y_metricos.hypervolume import calculate_hypervolume
from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import calculate_average_ratio_pareto, find_pareto_front, inverted_generational_distance, spacing_metric, update_population, visualize_pareto_clustering
from convergencia_iteraciones_mrl_amis.analisis_convergencia import run_convergence_analysis
from ejecucion_iteraciones_mrl_amis.ejecutar_mrl_amis import run_mrl_amis_for_multiple_groups
from estado_y_recompensa_rl.definir_comportamiento import calculate_reward, get_state
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
from generacionWorkPackages.workPackages import decodificar_wp, generar_work_packages
from intelligence_boxes.definir_intelligence_boxes import (
    ib_random_perturbation,
    ib_swap_mutation,
    ib_inversion_mutation,
    ib_guided_perturbation,
    ib_local_search,
    ib_diversity_mutation
)
from rutas_optimas.creacion_rutas_en_mapas import create_realistic_route_maps
from rutas_optimas.identificacion_topsis import identify_optimal_routes_topsis

def generate_data(num_pois: int = 15, seed: int = 42):
    """
    Genera datos sintéticos para MRL-AMIS usando programación orientada a objetos.
    Reemplaza cargar_datos() para no depender de archivos Excel.
    
    Args:
        num_pois: Número de POIs a generar (default: 15)
        seed: Semilla para reproducibilidad (default: 42)
        
    Returns:
        Dict con la misma estructura que cargar_datos() pero con datos generados sintéticamente
    """
    print(f"Generando datos sintéticos con {num_pois} POIs...")
    
    # Generar todos los datos usando el generador OOP
    data = generate_synthetic_data(num_pois=num_pois, seed=seed)
    
    # Verificar que las estructuras de datos estén correctas
    required_keys = [
        'pois', 'tourist_groups', 'distances', 'travel_times', 'costs',
        'co2_emission_cost', 'accident_risk', 'preferencias_grupos_turistas', 
        'costos_experiencia', 'parametros_generales'
    ]
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"Advertencia: Faltan claves en los datos: {missing_keys}")
    
    # Aplicar el mismo procesamiento que cargar_datos() para compatibilidad
    if "preferencias_grupos_turistas" in data:
        prefs = data["preferencias_grupos_turistas"]
        # Asegurar que los índices tengan el formato correcto
        prefs.index = [f"grupo_{idx}" if not str(idx).startswith("grupo_") else str(idx) for idx in prefs.index]
        data["preferencias_grupos_turistas"] = prefs
    
    # Verificar valores nulos
    for name, df in data.items():
        if name != "parametros_generales" and hasattr(df, "isnull"):
            if df.isnull().values.any():
                print(f"Advertencia: {name} contiene valores nulos")
    
    print("Datos sintéticos generados y limpios.")
    return data

def ejecutar_modelo_con_datos_sinteticos():
    # OPCIÓN 1: Usar datos sintéticos (recomendado - no necesita Excel)
    print("Generando datos sintéticos para MRL-AMIS...")
    data = generate_data(num_pois=15)
    
    # OPCIÓN 2: Usar datos desde Excel (comentado por defecto)
    # print("Cargando datos desde archivos Excel...")
    # data = cargar_datos()
    
    print("Datos cargados:")
    
    # DEBUG: Imprimir todas las matrices y datos generados
    print("\n" + "="*50)
    print("DEBUG: DATOS GENERADOS")
    print("="*50)
    
    # Imprimir POIs
    print("\n--- POIs GENERADOS ---")
    print(data['pois'])
    print(f"\nColumnas POIs: {list(data['pois'].columns)}")
    print(f"Índices POIs: {list(data['pois'].index)}")
    
    # Imprimir grupos turísticos
    print("\n--- GRUPOS TURÍSTICOS ---")
    print(data['tourist_groups'])
    print(f"\nColumnas Grupos: {list(data['tourist_groups'].columns)}")
    print(f"Índices Grupos: {list(data['tourist_groups'].index)}")
    
    # Imprimir matriz de distancias
    print("\n--- MATRIZ DE DISTANCIAS ---")
    print(data['distances'])
    
    # Imprimir matriz de tiempos de viaje
    print("\n--- MATRIZ DE TIEMPOS DE VIAJE ---")
    print(data['travel_times'])
    
    # Imprimir matriz de costos
    print("\n--- MATRIZ DE COSTOS ---")
    print(data['costs'])
    
    # Imprimir matriz de CO2
    print("\n--- MATRIZ DE CO2 ---")
    print(data['co2_emission_cost'])
    
    # Imprimir matriz de riesgo de accidentes
    print("\n--- MATRIZ DE RIESGO DE ACCIDENTES ---")
    print(data['accident_risk'])
    
    # Imprimir preferencias de grupos
    print("\n--- PREFERENCIAS DE GRUPOS ---")
    print(data['preferencias_grupos_turistas'])
    
    # Imprimir costos de experiencias
    print("\n--- COSTOS DE EXPERIENCIAS ---")
    print(data['costos_experiencia'])
    print(f"\nColumnas Costos Experiencia: {list(data['costos_experiencia'].columns)}")
    
    # Imprimir parámetros generales
    print("\n--- PARÁMETROS GENERALES ---")
    print(data['parametros_generales'])
    
    print("\n" + "="*50)
    print("FIN DEBUG")
    print("="*50 + "\n")
    # 2. Parámetros del Algoritmo

    num_work_packages = 100 # Número de Work Packages (soluciones)
    num_pois = 15 # Número total de POIs disponibles
    max_pois_per_route = 10 # Máximo número de POIs por ruta (ajustar según sea necesario)
    min_pois_per_route = 3 # Mínimo número de POIs por ruta (ajustar según sea necesario)
    num_routes_per_wp = 3 # Número de rutas por Work Package (ajustar según sea necesario)
    #max_distance_per_route = 500 # Ejemplo: Máxima distancia total permitida por ruta (en km)
    max_duration_per_route = 720 # Ejemplo: Máxima duración total permitida por ruta (en minutos)
    # Límites o metas para los objetivos de sostenibilidad, riesgo, etc.
    maximize_objectives_list = [True, False, False, True, False] # [Preference (MAX), Cost (MIN), CO2 (MIN), Combined Sustainability (MAX), Risk (MIN)]

    # Parámetros para el cálculo del Hypervolume (ejemplo, ajustar al espacio de objetivos)
    # Punto de referencia: debe ser peor que el peor de los valores posibles en cada objetivo
    ref_point_hypervolume = [0, 1000, 1000, -1000, 1000] # [Preference (min 0), Cost (max), CO2 (max), Sustainability (min), Risk (max)]
    # Nota: El Hypervolume se maximiza, por lo que los objetivos a minimizar deben ser manejados adecuadamente (ej. multiplicando por -1 o definiendo el punto de referencia correctamente)

    grupo_info_ejemplo = {
        'id': 'Grupo_2',
        'tiempo_disponible': 480, # 8 horas en minutos
        'presupuesto': 500, # Ejemplo de presupuesto expresado en USD
        'intereses': ['historia', 'cultura'], # Ejemplo de intereses
        'origen': '1', # ID del POI de origen (hotel, etc.)
        'min_pois_per_route': min_pois_per_route,
        'max_pois_per_route': max_pois_per_route
    }

    # Parámetros
    Q = 100  # número de soluciones iniciales
    D = len(data["pois"])  # D = 15  
    # Crear los Work Packages iniciales
    wps_df = generar_work_packages(q=num_work_packages, d=num_pois)
    print(f"Generados {len(wps_df)} Work Packages iniciales de dimensión {wps_df.shape[1]}") 

    # Este bloque decodifica y evalúa la población inicial de Work Packages (wps_df) para generar los resultados iniciales para el bucle MRL-AMIS.

    # Paso 1: Seleccionar el grupo turístico (ejemplo: grupo_3)
    try:
        grupo_id = 'grupo_3'  # Se puede seleccionar cualquiera de los grupos turísticos creados

        # Crear grupo_info con información necesaria
        grupo_info = {
            "tiempo_disponible": data["tourist_groups"].loc[grupo_id, "tiempo_max_min"],
            'min_pois_per_route': data["tourist_groups"].loc[grupo_id, "min_pois"]
                if 'min_pois' in data["tourist_groups"].columns else 0,
            'max_pois_per_route': data["tourist_groups"].loc[grupo_id, "max_pois"]
                if 'max_pois' in data["tourist_groups"].columns else float('inf'),
            'origen': str(data["tourist_groups"].loc[grupo_id, "origen"])
                if 'origen' in data["tourist_groups"].columns
                and not pd.isnull(data["tourist_groups"].loc[grupo_id, "origen"]) else '1'
        }

        tipo_turista = data["tourist_groups"].loc[grupo_id, "tipo_turista"]
        origen_poi = grupo_info['origen']

        print(f"Grupo turístico seleccionado: {grupo_id}, tipo: {tipo_turista}, origen: {origen_poi}")
        print(f"Tiempo disponible: {grupo_info['tiempo_disponible']} min")
        print(f"Restricciones: min_pois={grupo_info['min_pois_per_route']}, "
            f"max_pois={grupo_info['max_pois_per_route']}")

    except Exception as e:
        print(f"Error al seleccionar datos del grupo turístico: {e}. Usando valores por defecto.")
        grupo_info = {
            'tiempo_disponible': 720,  # Increased to match max_duration_per_route
            'min_pois_per_route': 2,   # Set minimum POIs to ensure some route structure
            'max_pois_per_route': 8,   # Set reasonable maximum
            'origen': '1'
        }
        tipo_turista = "nacional"
        origen_poi = grupo_info['origen']

    # Paso 2: Decodificar y Evaluar los 100 WPs iniciales
    resultados_wp_iniciales = []

    # DEBUG: Verificar WPs generados
    print(f"\n--- DEBUG: WORK PACKAGES GENERADOS ---")
    print(f"Número de WPs: {len(wps_df)}")
    print(f"Dimensión de cada WP: {len(wps_df.columns)}")
    print("Primeros 3 WPs:")
    print(wps_df.head(3))
    print(f"Grupo info usado para decodificación: {grupo_info}")
    print(f"Origen POI: {origen_poi}")

    # Verificar que las estructuras de datos necesarias estén disponibles
    if all(key in data for key in ["pois", "travel_times"]):
        print(f"\nDecodificando y evaluando {len(wps_df)} Work Packages iniciales...")

        # Variables para seguimiento de progreso
        start_time = time.time()
        num_factibles = 0
        debug_count = 0

        # Procesar cada WP
        for wp_name, wp_series in wps_df.iterrows():
                wp_vector = wp_series.values

                # DEBUG: Mostrar los primeros 3 WPs con detalle
                if debug_count < 3:
                    print(f"\n--- DEBUG WP {wp_name} ---")
                    print(f"Vector WP: {wp_vector}")

                # Decodificar WP → ruta y factibilidad
                ruta_decodificada, is_feasible = decodificar_wp(
                    wp_vector=wp_vector,
                    pois_df=data["pois"],
                    travel_times_df=data["travel_times"],
                    grupo_info=grupo_info,
                    origen=origen_poi
                )

                # DEBUG: Mostrar decodificación de los primeros 3 WPs
                if debug_count < 3:
                    print(f"Ruta decodificada: {ruta_decodificada}")
                    print(f"Es factible: {is_feasible}")

                # Evaluar ruta - métricas
                metrics = evaluar_funciones_objetivo(ruta_decodificada, data, tipo_turista=tipo_turista)

                # DEBUG: Mostrar métricas de los primeros 3 WPs
                if debug_count < 3:
                    print(f"Métricas: {metrics}")
                    debug_count += 1

                # Actualizar contador de soluciones factibles
                if is_feasible:
                    num_factibles += 1

                # Agregar WP, ruta, métricas y factibilidad al resultado
                resultados_wp_iniciales.append({
                    "wp_name": wp_name,
                    "wp_original": wp_vector,
                    "ruta_decodificada": ruta_decodificada,
                    "objetivos": metrics,
                    "is_feasible": is_feasible
                })

                # Mostrar progreso cada 10 WPs
                if len(resultados_wp_iniciales) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Procesados {len(resultados_wp_iniciales)}/{len(wps_df)} WPs "
                        f"({num_factibles} factibles) - Tiempo: {elapsed:.2f}s")

            # Resumen final
        elapsed = time.time() - start_time
        print(f"\nProcesamiento completado en {elapsed:.2f} segundos")
        print(f"Total de soluciones factibles: {num_factibles}/{len(wps_df)} "
        f"({num_factibles/len(wps_df)*100:.1f}%)")

            # Calcular estadísticas de objetivos para soluciones factibles
        if num_factibles > 0:
            objetivos_factibles = {
                "preferencia_total": [],
                "costo_total": [],
                "co2_total": [],
                "sust_ambiental": [],
                "sust_economica": [],
                "sust_social": [],
                "riesgo_total": []
            }

            for sol in resultados_wp_iniciales:
                if sol['is_feasible']:
                    for key in objetivos_factibles:
                        objetivos_factibles[key].append(sol['objetivos'].get(key, 0))

            print("\nEstadísticas de objetivos para soluciones factibles:")
            for key, values in objetivos_factibles.items():
                print(f"{key}: Min={min(values):.2f}, Max={max(values):.2f}, "
                    f"Promedio={sum(values)/len(values):.2f}")

    else:
        print("Error: Dataframes necesarios no están en el diccionario 'data'.")

    # Resultados_wp_iniciales contiene todos los WPs evaluados
    # Esto será utilizado como población inicial para MRL-AMIS

    # Opcional: Convertir a DataFrame para inspección
    df_resultados_iniciales = pd.DataFrame([
        {
            "wp_name": sol["wp_name"],
            "factible": sol["is_feasible"],
            "num_pois": len(sol["ruta_decodificada"]) - 2,  # Restar origen y destino
            "preferencia": sol["objetivos"].get("preferencia_total", 0),
            "costo": sol["objetivos"].get("costo_total", 0),
            "co2": sol["objetivos"].get("co2_total", 0),
            "riesgo": sol["objetivos"].get("riesgo_total", 0)
        }
        for sol in resultados_wp_iniciales
    ])

    # Mostrar resumen
    if not df_resultados_iniciales.empty:
        print("\nResumen de resultados iniciales:")
        print(f"Total de soluciones: {len(df_resultados_iniciales)}")
        print(f"Soluciones factibles: {df_resultados_iniciales['factible'].sum()}")
        print("\nEstadísticas de soluciones factibles:")
        feasible_results = df_resultados_iniciales[df_resultados_iniciales['factible']]
        if not feasible_results.empty:
            print(feasible_results.describe())
        else:
            print("No hay soluciones factibles para mostrar estadísticas.")

        # Definir el conjunto de Intelligence Boxes disponibles (acciones para el agente RL).
    intelligence_boxes = {
        0: ib_random_perturbation,
        1: ib_swap_mutation,
        2: ib_inversion_mutation,
        3: ib_guided_perturbation,
        4: ib_local_search,
        5: ib_diversity_mutation  # El nuevo operador
    }

    num_actions = 6

    print(f"Definidos {num_actions} Intelligence Boxes para MRL-AMIS")

    # Set proper state space and action space sizes
    state_space_size = 7  # Based on get_state function
    # num_actions is already set to 6 above

    # Parámetros ajustados para el agente RL
    rl_learning_rate = 0.2
    rl_discount_factor = 0.85
    rl_epsilon_start = 1.0
    rl_epsilon_decay_rate = 0.995
    rl_min_epsilon = 0.15

    # Creación del agente RL con parámetros ajustados
    rl_agent = QLearningAgent(
        state_space_size=7,
        action_space_size=6,
        learning_rate=rl_learning_rate,
        discount_factor=rl_discount_factor,
        epsilon=rl_epsilon_start,
        epsilon_decay_rate=rl_epsilon_decay_rate,
        min_epsilon=rl_min_epsilon
    )

    print("\nAgente QLearning inicializado con parámetros optimizados.")

    # Este es el bucle principal que impulsa el algoritmo MRL-AMIS.
    # Integra el agente RL, las cajas de inteligencia, la decodificación, la evaluación y la actualización de la población

    # Inicialización antes del bucle
    hypervolume_history = []  # Lista para guardar el historial de hipervolumen

    # --- Parámetros del bucle principal ---
    max_iterations = 100 # Número total de iteraciones para el algoritmo MRL-AMIS

    # --- Inicialización antes del bucle ---
    # Comenzar la población actual con los resultados de la evaluación inicial
    if resultados_wp_iniciales:
        current_population_results = resultados_wp_iniciales.copy()
        print(f"Inicializando bucle con {len(current_population_results)} Work Packages iniciales")
    else:
        print("Advertencia: No se encontraron Work Packages iniciales válidos.")
        current_population_results = []

    # Set default values for missing parameters
    maximize_objectives_list = [True, False, False, True, False]  # [Preferencia+, Costo-, CO2-, Sostenibilidad+, Riesgo-]
    num_work_packages = 100
    ref_point_hypervolume = [1, 1, 1, 1, 1]  # Default reference point

    # Ensure required variables are initialized
    if num_actions == 0:
        print("Advertencia: num_actions es 0, esto puede causar problemas en el bucle")
        
    print(f"Configuración del bucle - WPs: {num_work_packages}, Max iter: {max_iterations}")


    # --- Bucle iterativo principal MRL-AMIS ---
    print("\nStarting MRL-AMIS iterative process...")

    # --- Bucle iterativo principal MRL-AMIS ---
    print("\nStarting MRL-AMIS iterative process...")

    # Estructuras de datos para almacenar métricas y resultados por iteración
    iteration_metrics = {
        'iteration': [],
        'pareto_front_size': [],
        'hypervolume': [],
        'arp': [],
        'spacing': [],
        'execution_time': []
    }

    start_time_total = time.time()

    for iteration in range(max_iterations):
        iter_start_time = time.time()
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        generated_solutions_this_iteration = []

        # Extraer los objetivos y los datos de viabilidad de la población actual
        current_population_objectives_dicts = [sol.get('objetivos', {}) for sol in current_population_results]
        current_population_feasibility = [sol.get('is_feasible', False) for sol in current_population_results]

        # Convertir los objetivos de población en una matriz numpy
        current_population_objectives_np_all = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
            obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0), # Combined Sustainability
            obj_dict.get('riesgo_total', 0)]
            for obj_dict in current_population_objectives_dicts
        ])

        # Encontrar el frente de Pareto actual
        current_feasible_indices = [i for i, is_feat in enumerate(current_population_feasibility) if is_feat]
        current_feasible_objectives_np = current_population_objectives_np_all[current_feasible_indices] if current_feasible_indices else np.array([])

        current_pareto_front_objectives = []
        if len(current_feasible_objectives_np) > 0:
            current_pareto_front_objectives, _ = find_pareto_front(current_feasible_objectives_np, maximize_objectives_list)

        # --- Aplicar Intelligence Boxes para generar nuevas soluciones ---
        for original_wp_result in current_population_results:
            original_wp_vector_dict = original_wp_result.get('wp_original', {})

            # Convertir wp_original en una matriz numpy
            if isinstance(original_wp_vector_dict, dict) and original_wp_vector_dict:
                original_wp_vector = np.array(list(original_wp_vector_dict.values()))
            elif isinstance(original_wp_result.get('wp_original'), (np.ndarray, pd.Series)):
                original_wp_vector = original_wp_result['wp_original']
            else:
                original_wp_vector = np.zeros(len(original_wp_result.get('wp_original', [])))

            # --- Decisión RL ---
            # Obtener estado actual
            current_state = get_state(original_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)

            # El agente RL elige una caja de inteligencia
            if num_actions > 0:
                action_index = rl_agent.choose_action(current_state)
                if action_index < len(intelligence_boxes):
                    chosen_intelligence_box_func = intelligence_boxes[action_index]
                else:
                    print(f"Warning: RL agent chose invalid action index {action_index}. Clamping to 0.")
                    chosen_intelligence_box_func = intelligence_boxes[0]
                    action_index = 0
            else:
                print("Error: No Intelligence Boxes defined. Skipping operator application for this WP.")
                continue

            # Preparar kwargs con todos los parámetros posibles que podrían necesitar los IBs
            kwargs = {
                'grupo_info': grupo_info_ejemplo,
                'pois_df': data["pois"],
                'travel_times_df': data["travel_times"],
                'data': data,
                'tipo_turista': tipo_turista,
                'maximize_objectives_list': maximize_objectives_list,
                'current_pareto_front': current_pareto_front_objectives,
                'current_population_results': current_population_results,
                'current_pareto_front_indices': current_feasible_indices,  # Añadir los índices
                'origen': '1'
            }

            # Aplicar Intelligence Box con interfaz estandarizada
            modified_wp_vector = chosen_intelligence_box_func(original_wp_vector, **kwargs)

            # Descodificar el WP modificado en una ruta y obtener el estado de viabilidad.
            # SOLUCIÓN: Descomentamos y adaptamos esta parte
            decoded_route, is_feasible = decodificar_wp(
                wp_vector=modified_wp_vector,
                pois_df=data["pois"],
                travel_times_df=data["travel_times"],
                grupo_info=grupo_info_ejemplo,
                origen='1'
            )

            # Evaluar la ruta descodificada
            evaluated_metrics = evaluar_funciones_objetivo(decoded_route, data, tipo_turista)

            # Almacenar el resultado para el WP modificado
            modified_wp_result = {
                'wp_original': modified_wp_vector,
                'ruta_decodificada': decoded_route,
                'objetivos': evaluated_metrics,
                'is_feasible': is_feasible
            }

            # Calcular recompensa
            reward = calculate_reward(original_wp_result, modified_wp_result, current_pareto_front_objectives, maximize_objectives_list)

            # Determinar el siguiente estado
            next_state = get_state(modified_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)

            # El agente RL aprende de la experiencia
            rl_agent.learn(current_state, action_index, reward, next_state)

            # Añadir WP modificado a las soluciones generadas
            generated_solutions_this_iteration.append(modified_wp_result)

        # Decaimiento épsilon para exploración
        rl_agent.decay_epsilon_with_restart()

        # --- Actualización de la población (selección ambiental) ---
        current_population_results = update_population(
            current_population_results,
            generated_solutions_this_iteration,
            num_work_packages,
            maximize_objectives_list
        )

        # --- Analizar e informar sobre el progreso ---
        # Extraer objetivos de la población actualizada (solo soluciones viables)
        objectives_updated_pop_feasible = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
            obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
            obj_dict.get('riesgo_total', 0)]
            for sol in current_population_results
            if sol.get('is_feasible', False)
            for obj_dict in [sol.get('objetivos', {})]
            if obj_dict
        ])

        # Calcular métricas para esta iteración
        iteration_hv = 0.0
        iteration_arp = 0.0
        iteration_spacing = 0.0
        iteration_pareto_size = 0

        if len(objectives_updated_pop_feasible) > 0:
            # Encuentre el frente de Pareto para esta iteración
            current_pareto_front_objectives_iter, pareto_indices = find_pareto_front(
                objectives_updated_pop_feasible,
                maximize_objectives_list
            )
            iteration_pareto_size = len(current_pareto_front_objectives_iter)

            # Calcular métricas si tenemos un frente de Pareto
            if iteration_pareto_size > 0:
                try:
                    # Calcular Hypervolume
                    iteration_hv = calculate_hypervolume(
                        current_pareto_front_objectives_iter,
                        ref_point_hypervolume,
                        maximize_objectives_list
                    )

                    hypervolume_history.append(iteration_hv)

                    # Calcular ARP
                    iteration_arp = calculate_average_ratio_pareto(
                        current_pareto_front_objectives_iter,
                        objectives_updated_pop_feasible,
                        maximize_objectives_list
                    )

                    # Calcular Spacing
                    iteration_spacing = spacing_metric(
                        current_pareto_front_objectives_iter,
                        maximize_objectives_list
                    )

                    print(f"Iteration {iteration + 1}: Pareto Front Size = {iteration_pareto_size}, "
                        f"HV = {iteration_hv:.4f}, ARP = {iteration_arp:.4f}, Spacing = {iteration_spacing:.4f}")

                except Exception as e:
                    print(f"Iteration {iteration + 1}: Error calculating metrics: {e}")
                    print(f"Iteration {iteration + 1}: Pareto Front Size = {iteration_pareto_size} (Metrics calculation error)")
            else:
                print(f"Iteration {iteration + 1}: No non-dominated solutions found in feasible population.")
        else:
            print(f"Iteration {iteration + 1}: No feasible solutions in population.")

        # Registrar el tiempo de ejecución de esta iteración
        iter_execution_time = time.time() - iter_start_time

        # Almacenar métricas para esta iteración
        iteration_metrics['iteration'].append(iteration + 1)
        iteration_metrics['pareto_front_size'].append(iteration_pareto_size)
        iteration_metrics['hypervolume'].append(iteration_hv)
        iteration_metrics['arp'].append(iteration_arp)
        iteration_metrics['spacing'].append(iteration_spacing)
        iteration_metrics['execution_time'].append(iter_execution_time)

        # Tiempo de iteración de impresión
        print(f"Iteration {iteration + 1} completed in {iter_execution_time:.2f} seconds")


    # --- Fin del bucle principal MRL-AMIS ---
    total_execution_time = time.time() - start_time_total
    print(f"\nMRL-AMIS iterative process finished in {total_execution_time:.2f} seconds.")


    # --- Análisis final y resultados ---
    # Extraer objetivos de la población final (solo soluciones viables)
    final_population_objectives_feasible = np.array([
        [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
        obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
        obj_dict.get('riesgo_total', 0)]
        for sol in current_population_results
        if sol.get('is_feasible', False)
        for obj_dict in [sol.get('objetivos', {})]
        if obj_dict
    ])

    # Encontrar el frente de Pareto final
    final_pareto_front_objectives = []
    final_pareto_front_indices = []

    if len(final_population_objectives_feasible) > 0:
        final_pareto_front_objectives, final_pareto_front_indices = find_pareto_front(
            final_population_objectives_feasible,
            maximize_objectives_list
        )

        # Obtener las soluciones reales correspondientes al frente de Pareto final
        final_feasible_solutions = [
            sol for sol in current_population_results
            if sol.get('is_feasible', False) and sol.get('objetivos') is not None and sol.get('objetivos')
        ]

        final_pareto_front_solutions = [final_feasible_solutions[i] for i in final_pareto_front_indices]

    # Análisis de diversidad del frente de Pareto mediante clustering

    if len(final_pareto_front_solutions) >= 5:  # Necesitamos al menos 5 soluciones para clustering
        print("\nAnalizando diversidad del frente de Pareto con clustering...")
        visualize_pareto_clustering(final_pareto_front_solutions, final_pareto_front_objectives, maximize_objectives_list)
    else:
        print("\nNo hay suficientes soluciones en el frente de Pareto para análisis de clustering (mínimo 5 requeridas).")


        # Calcular las métricas finales
        final_hv = calculate_hypervolume(
            final_pareto_front_objectives,
            ref_point_hypervolume,
            maximize_objectives_list
        )

        final_arp = calculate_average_ratio_pareto(
            final_pareto_front_objectives,
            final_population_objectives_feasible,
            maximize_objectives_list
        )

        final_spacing = spacing_metric(
            final_pareto_front_objectives,
            maximize_objectives_list
        )

        # Imprimir resultados finales
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Final Pareto Front Size: {len(final_pareto_front_objectives)}")
        print(f"Final Hypervolume: {final_hv:.4f}")
        print(f"Final ARP: {final_arp:.4f}")
        print(f"Final Spacing: {final_spacing:.4f}")

        # IGD calculation would require a reference front (not available in this context)
        print("IGD calculation skipped - no reference front available")

        # Imprimir rangos objetivos en el frente de Pareto
        print("\nObjective Ranges in Final Pareto Front:")
        obj_names = ["Preferencia Total", "Costo Total", "CO2 Total", "Sostenibilidad Combinada", "Riesgo Total"]
        obj_directions = ["MAX", "MIN", "MIN", "MAX", "MIN"]

        for i, (name, direction) in enumerate(zip(obj_names, obj_directions)):
            obj_values = final_pareto_front_objectives[:, i]
            print(f"{name} ({direction}): Min={np.min(obj_values):.2f}, Max={np.max(obj_values):.2f}, "
                f"Mean={np.mean(obj_values):.2f}, Std={np.std(obj_values):.2f}")

        # Soluciones de impresión con los mejores valores para cada objetivo
        print("\nBest Solutions for Each Objective:")
        for i, (name, direction) in enumerate(zip(obj_names, obj_directions)):
            if direction == "MAX":
                best_idx = np.argmax(final_pareto_front_objectives[:, i])
            else:
                best_idx = np.argmin(final_pareto_front_objectives[:, i])

            best_solution = final_pareto_front_solutions[best_idx]
            obj_values = final_pareto_front_objectives[best_idx]

            print(f"\nBest solution for {name} ({direction}):")
            print(f"  {name}: {obj_values[i]:.2f}")
            print(f"  Route: {best_solution['ruta_decodificada']}")
            print(f"  All Objectives: ", end="")
            for j, (obj_name, obj_val) in enumerate(zip(obj_names, obj_values)):
                print(f"{obj_name}={obj_val:.2f}", end=", " if j < len(obj_values)-1 else "")
            print()

        # Imprimir soluciones detalladas del frente de Pareto
        print("\nShow detailed Pareto front solutions? (y/n)")
        response = input().lower()
        if response == 'y':
            print("\nDetailed Pareto Front Solutions:")
            for i, solution in enumerate(final_pareto_front_solutions):
                print(f"\nSolution {i+1}:")
                print(f"  Route: {solution['ruta_decodificada']}")
                print(f"  Objectives: {solution['objetivos']}")
        else:
            print("\nNo feasible solutions found in the final population.")


    # --- Análisis de la evolución de las métricas ---
    print("\n" + "="*50)
    print("METRICS EVOLUTION")
    print("="*50)

    
    # --- Análisis del agente RL ---
    print("\n" + "="*50)
    print("RL AGENT ANALYSIS")
    print("="*50)

    # Gráfico de valores Q
    if hasattr(rl_agent, 'q_table'):
        # Gráfico del valor Q promedio para cada estado
        avg_q_values = np.mean(rl_agent.q_table, axis=1)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_q_values)), avg_q_values)
        plt.xlabel('State')
        plt.ylabel('Average Q-Value')
        plt.title('Average Q-Value per State')
        plt.grid(True)
        plt.savefig('rl_avg_q_values.png')
        plt.close()

        print("Average Q-values per state plot saved as 'rl_avg_q_values.png'")

        # Gráfico de preferencias de acción para cada estado
        fig, axs = plt.subplots(min(6, len(rl_agent.q_table)), 1, figsize=(10, 10), sharex=True)

        # Si solo hay un estado, crea una lista axs para gestionarlo de forma coherente
        if len(rl_agent.q_table) == 1:
            axs = [axs]

        for i, state_q_values in enumerate(rl_agent.q_table[:min(6, len(rl_agent.q_table))]):
            axs[i].bar(range(len(state_q_values)), state_q_values)
            axs[i].set_ylabel(f'Q-Value')
            axs[i].set_title(f'State {i} Q-Values')
            axs[i].grid(True)

        plt.xlabel('Action (Intelligence Box)')
        plt.tight_layout()
        plt.savefig('rl_state_action_values.png')
        plt.close()

        print("State-action Q-values plot saved as 'rl_state_action_values.png'")

    # Imprimir estadísticas del agente RL
    print("\nRL Agent Statistics:")
    if hasattr(rl_agent, 'epsilon'):
        print(f"Final Epsilon: {rl_agent.epsilon:.4f}")

    if hasattr(rl_agent, 'action_counts'):
        print("\nAction (Intelligence Box) Usage:")
        total_actions = sum(rl_agent.action_counts)
        for i, count in enumerate(rl_agent.action_counts):
            print(f"  IB {i}: {count} uses ({count/total_actions*100:.2f}%)")

    if hasattr(rl_agent, 'state_visits'):
        print("\nState Visit Counts:")
        total_visits = sum(rl_agent.state_visits)
        for i, count in enumerate(rl_agent.state_visits):
            print(f"  State {i}: {count} visits ({count/total_visits*100:.2f}%)")


    # --- Guardar resultados para uso futuro ---
    import pickle

    # Crear un diccionario con todos los resultados importantes
    results_data = {
        'iteration_metrics': iteration_metrics,
        'final_pareto_front_objectives': final_pareto_front_objectives if 'final_pareto_front_objectives' in locals() else None,
        'final_pareto_front_solutions': final_pareto_front_solutions if 'final_pareto_front_solutions' in locals() else None,
        'execution_time': total_execution_time,
        'maximize_objectives_list': maximize_objectives_list,
        'run_timestamp': time.strftime("%Y%m%d-%H%M%S")
    }

    # Guardar los resultados
    with open('mrl_amis_results.pickle', 'wb') as f:
        pickle.dump(results_data, f)

    print("\nResults saved to 'mrl_amis_results.pickle' for future analysis.")
    print("\nMRL-AMIS execution completed successfully!")

    # Definir grupos para análisis detallado
    analysis_groups = ['grupo_1', 'grupo_2', 'grupo_3', 'grupo_4', 'grupo_5', 'grupo_6']
    iterations_list = [100]

    # Ejecutar análisis detallado
    results_detailed = run_mrl_amis_for_multiple_groups(
        data=data,
        groups_to_process=analysis_groups,
        max_iterations_list=iterations_list,
        num_work_packages=num_work_packages,
        ref_point_hypervolume=ref_point_hypervolume,    
        maximize_objectives_list=maximize_objectives_list,
    )

    # Analizar la convergencia del ARP para todos los grupos
    print("\n" + "="*60)
    print("ANÁLISIS DE CONVERGENCIA DEL ARP PARA TODOS LOS GRUPOS")
    print("="*60)

    arp_summary = {}
    for grupo_id in analysis_groups:
        print(f"\n{grupo_id.upper()}:")
        if grupo_id in results_detailed:
            arp_values = []

            for iterations in iterations_list:
                if iterations in results_detailed[grupo_id]:
                    result = results_detailed[grupo_id][iterations]

                    if len(result['final_pareto_front_objectives']) > 0:
                        # Calcular población completa para ARP
                        all_objectives = np.array([
                            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0),
                            obj_dict.get('co2_total', 0),
                            obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) +
                            obj_dict.get('sust_social', 0),
                            obj_dict.get('riesgo_total', 0)]
                            for sol in result['current_population_results']
                            for obj_dict in [sol.get('objetivos', {})]
                            if obj_dict
                        ])

                        arp = calculate_average_ratio_pareto(
                            result['final_pareto_front_objectives'],
                            all_objectives,
                            maximize_objectives_list
                        )
                        arp_values.append((iterations, arp))
                        print(f"  {iterations} iteraciones: ARP = {arp:.4f}")
                    else:
                        arp_values.append((iterations, 0))
                        print(f"  {iterations} iteraciones: ARP = 0.0000 (sin frente de Pareto)")

            # Determinar el mejor punto según ARP
            if arp_values:
                best_iterations, best_arp = max(arp_values, key=lambda x: x[1])
                print(f"  → Mejor resultado: {best_iterations} iteraciones con ARP = {best_arp:.4f}")
                arp_summary[grupo_id] = {
                    'values': arp_values,
                    'best_iterations': best_iterations,
                    'best_arp': best_arp
                }

    # Crear tabla resumen del ARP para todos los grupos
    print("\n" + "="*60)
    print("TABLA RESUMEN DE ARP")
    print("="*60)

    # Crear DataFrame para analizar
    summary_data = []
    for grupo_id in analysis_groups:
        if grupo_id in arp_summary:
            for iterations, arp in arp_summary[grupo_id]['values']:
                summary_data.append({
                    'Grupo': grupo_id,
                    'Iteraciones': iterations,
                    'ARP': arp,
                    'Es_Optimo': iterations == arp_summary[grupo_id]['best_iterations']
                })

    summary_df = pd.DataFrame(summary_data)

    # Mostrar tabla pivot
    pivot_table = summary_df.pivot(index='Grupo', columns='Iteraciones', values='ARP')
    print("\nTabla ARP por Grupo e Iteraciones:")
    print(pivot_table.round(4))

    # Guardar en Excel
    summary_df.to_excel('analisis_arp_convergencia.xlsx', index=False)
    print(f"\nAnálisis de ARP guardado en 'analisis_arp_convergencia.xlsx'")

    # Identificar recomendación global
    print("\n" + "="*60)
    print("RECOMENDACIONES GLOBALES")
    print("="*60)

    convergence_analysis = summary_df.groupby('Iteraciones')['ARP'].agg(['mean', 'std', 'min', 'max'])
    print("\nEstadísticas del ARP por número de iteraciones:")
    print(convergence_analysis.round(4))

    # Identificar punto de convergencia general
    mean_arp_by_iterations = summary_df.groupby('Iteraciones')['ARP'].mean()
    improvements = []
    for i in range(1, len(mean_arp_by_iterations)):
        current_iter = mean_arp_by_iterations.index[i]
        prev_iter = mean_arp_by_iterations.index[i-1]
        improvement = ((mean_arp_by_iterations[current_iter] - mean_arp_by_iterations[prev_iter]) /
                    max(mean_arp_by_iterations[prev_iter], 0.0001)) * 100
        improvements.append((current_iter, improvement))

    print("\nAnálisis de mejoras promedio en ARP:")
    for iterations, improvement in improvements:
        print(f"De {improvements[improvements.index((iterations, improvement))-1][0] if improvements.index((iterations, improvement)) > 0 else iterations_list[0]} a {iterations} iteraciones: +{improvement:.2f}%")

    # Determinar si hay convergencia general
    convergence_threshold = 1.0  # 1% de mejora como umbral
    global_convergence = None
    for i, (iterations, improvement) in enumerate(improvements):
        if all(imp[1] < convergence_threshold for imp in improvements[i:]):
            global_convergence = improvements[i-1][0] if i > 0 else iterations_list[0]
            break

    if global_convergence:
        print(f"\nRECOMENDACIÓN GLOBAL: El ARP converge aproximadamente en {global_convergence} iteraciones.")
        print(f"Mejora adicional después de {global_convergence} iteraciones es marginal.")
    else:
        print(f"\nRECOMENDACIÓN GLOBAL: El ARP continúa mejorando hasta 1000 iteraciones.")
        print("Considera aumentar el número de iteraciones si se requiere mayor precisión.")

    # Para ejecutar el análisis
    # Definir número de iteraciones a probar
    iterations_list = [100, 200, 400, 600, 800, 1000]

    # Ejecutar análisis de convergencia para grupo_2
    convergence_results = run_convergence_analysis(
        data=data,
        grupo_id='grupo_2',
        iterations_list=iterations_list,
        num_work_packages=num_work_packages,
        ref_point_hypervolume=ref_point_hypervolume,
        maximize_objectives_list=maximize_objectives_list
    ) 

    # Ejemplo de uso
    groups_to_process = ['grupo_1', 'grupo_2', 'grupo_3', 'grupo_4', 'grupo_5', 'grupo_6']

    optimal_routes = identify_optimal_routes_topsis(
        data=data,
        groups_to_process=groups_to_process,
        iterations_for_analysis=1000, # Usar 1000 iteraciones para un análisis completo
        maximize_objectives_list=maximize_objectives_list,
        num_work_packages=num_work_packages,
    )

    if 'latitud' in data['pois'].columns and 'longitud' in data['pois'].columns:
        data['pois']['latitude'] = data['pois']['latitud']
        data['pois']['longitude'] = data['pois']['longitud']
        print("Added latitude/longitude aliases for map creation")

    if 'lat' in data['pois'].columns and 'lng' in data['pois'].columns:
        data['pois']['latitud'] = data['pois']['lat']
        data['pois']['longitude'] = data['pois']['lng']
        print("Added latitude/longitude aliases for map creation")



    # Ejemplo de uso después de calcular las rutas óptimas
    create_realistic_route_maps(
        optimal_routes=optimal_routes,
        data=data,
        selected_groups=['grupo_2', 'grupo_3'],
        top_n=3
    )

    # Return the complete results
    return results_data
