from AgenteQLearning.QLearningAgent import QLearningAgent
from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import calculate_average_ratio_pareto, calculate_hypervolume, find_pareto_front, spacing_metric, update_population
from estado_y_recompensa_rl.definir_comportamiento import calculate_reward, get_state
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
from generacionWorkPackages.workPackages import decodificar_wp, generar_work_packages
from intelligence_boxes.definir_intelligence_boxes import ib_diversity_mutation, ib_guided_perturbation, ib_inversion_mutation, ib_local_search, ib_random_perturbation, ib_swap_mutation
import numpy as np
import pandas as pd
import time

def execute_mrl_amis(resultados_wp_iniciales, data, grupo_info, tipo_turista,
                     max_iterations, num_work_packages, intelligence_boxes=None,
                     maximize_objectives_list=None, ref_point_hypervolume=None):
    """
    Ejecuta el algoritmo MRL-AMIS con los parámetros dados.

    Returns:
        dict: Resultados de la ejecución incluyendo métricas y soluciones finales
    """

    # Usar valores por defecto si no se proporcionan
    if intelligence_boxes is None:
        intelligence_boxes = {
            0: ib_random_perturbation,
            1: ib_swap_mutation,
            2: ib_inversion_mutation,
            3: ib_guided_perturbation,
            4: ib_local_search,
            5: ib_diversity_mutation
        }

    if maximize_objectives_list is None:
        maximize_objectives_list = [True, False, False, True, False]

    if ref_point_hypervolume is None:
        ref_point_hypervolume = [0, 1000, 1000, -1000, 1000]

    # Inicializar agente RL
    rl_agent = QLearningAgent(
        state_space_size=7,
        action_space_size=len(intelligence_boxes),
        learning_rate=0.2,
        discount_factor=0.85,
        epsilon=1.0,
        epsilon_decay_rate=0.995,
        min_epsilon=0.15
    )

    # Inicializar estructuras de datos
    current_population_results = resultados_wp_iniciales.copy()
    hypervolume_history = []
    iteration_metrics = {
        'iteration': [],
        'pareto_front_size': [],
        'hypervolume': [],
        'arp': [],
        'spacing': [],
        'execution_time': []
    }

    # Bucle principal MRL-AMIS
    start_time_total = time.time()

    print(f"\nIniciando MRL-AMIS para {grupo_info['id']}...")

    for iteration in range(max_iterations):
        iter_start_time = time.time()

        # Imprimir progreso cada 10 iteraciones
        if iteration % 10 == 0:
            print(f"  Iteración {iteration+1}/{max_iterations}...", end='\r')

        generated_solutions_this_iteration = []

        # Encontrar el frente de Pareto actual
        current_population_objectives_dicts = [sol.get('objetivos', {}) for sol in current_population_results]
        current_population_feasibility = [sol.get('is_feasible', False) for sol in current_population_results]

        current_population_objectives_np_all = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
             obj_dict.get('riesgo_total', 0)]
            for obj_dict in current_population_objectives_dicts
        ])

        current_feasible_indices = [i for i, is_feat in enumerate(current_population_feasibility) if is_feat]
        current_feasible_objectives_np = current_population_objectives_np_all[current_feasible_indices] if current_feasible_indices else np.array([])

        current_pareto_front_objectives = []
        if len(current_feasible_objectives_np) > 0:
            current_pareto_front_objectives, _ = find_pareto_front(current_feasible_objectives_np, maximize_objectives_list)

        # Aplicar Intelligence Boxes
        for original_wp_result in current_population_results:
            original_wp_vector_dict = original_wp_result.get('wp_original', {})

            if isinstance(original_wp_vector_dict, dict) and original_wp_vector_dict:
                original_wp_vector = np.array(list(original_wp_vector_dict.values()))
            elif isinstance(original_wp_result.get('wp_original'), (np.ndarray, pd.Series)):
                original_wp_vector = original_wp_result['wp_original']
            else:
                original_wp_vector = np.zeros(len(original_wp_result.get('wp_original', [])))

            # RL Agent selecciona una Intelligence Box
            current_state = get_state(original_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)
            action_index = rl_agent.choose_action(current_state)
            chosen_intelligence_box_func = intelligence_boxes[action_index]

            # Preparar kwargs
            kwargs = {
                'grupo_info': grupo_info,
                'pois_df': data["pois"],
                'travel_times_df': data["travel_times"],
                'data': data,
                'tipo_turista': tipo_turista,
                'maximize_objectives_list': maximize_objectives_list,
                'current_pareto_front': current_pareto_front_objectives,
                'current_population_results': current_population_results,
                'current_pareto_front_indices': current_feasible_indices,
                'origen': grupo_info['origen']
            }

            # Aplicar Intelligence Box
            modified_wp_vector = chosen_intelligence_box_func(original_wp_vector, **kwargs)

            # Decodificar y evaluar
            decoded_route, is_feasible = decodificar_wp(
                wp_vector=modified_wp_vector,
                pois_df=data["pois"],
                travel_times_df=data["travel_times"],
                grupo_info=grupo_info,
                origen=grupo_info['origen']
            )

            evaluated_metrics = evaluar_funciones_objetivo(decoded_route, data, tipo_turista)

            modified_wp_result = {
                'wp_original': modified_wp_vector,
                'ruta_decodificada': decoded_route,
                'objetivos': evaluated_metrics,
                'is_feasible': is_feasible
            }

            # Calcular recompensa y actualizar RL
            reward = calculate_reward(original_wp_result, modified_wp_result, current_pareto_front_objectives, maximize_objectives_list)
            next_state = get_state(modified_wp_result, current_pareto_front_objectives, maximize_objectives_list, hypervolume_history)
            rl_agent.learn(current_state, action_index, reward, next_state)

            generated_solutions_this_iteration.append(modified_wp_result)

        # Actualizar población
        current_population_results = update_population(
            current_population_results,
            generated_solutions_this_iteration,
            num_work_packages,
            maximize_objectives_list
        )

        # Calcular métricas para esta iteración
        objectives_updated_pop_feasible = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
             obj_dict.get('riesgo_total', 0)]
            for sol in current_population_results
            if sol.get('is_feasible', False)
            for obj_dict in [sol.get('objetivos', {})]
            if obj_dict
        ])

        # MODIFICACIÓN IMPORTANTE: Calcular también población completa para ARP
        objectives_updated_pop_all = np.array([
            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
             obj_dict.get('riesgo_total', 0)]
            for sol in current_population_results
            for obj_dict in [sol.get('objetivos', {})]
            if obj_dict
        ])

        iteration_hv = 0.0
        iteration_arp = 0.0
        iteration_spacing = 0.0
        iteration_pareto_size = 0

        if len(objectives_updated_pop_feasible) > 0:
            current_pareto_front_objectives_iter, pareto_indices = find_pareto_front(
                objectives_updated_pop_feasible,
                maximize_objectives_list
            )
            iteration_pareto_size = len(current_pareto_front_objectives_iter)

            if iteration_pareto_size > 0:
                try:
                    iteration_hv = calculate_hypervolume(
                        current_pareto_front_objectives_iter,
                        ref_point_hypervolume,
                        maximize_objectives_list
                    )
                    hypervolume_history.append(iteration_hv)

                    # MODIFICACIÓN IMPORTANTE: Calcular ARP correctamente
                    # ARP = (# soluciones no dominadas) / (# total de soluciones)
                    iteration_arp = calculate_average_ratio_pareto(
                        current_pareto_front_objectives_iter,  # Non-dominated solutions
                        objectives_updated_pop_all,            # Total population (all solutions)
                        maximize_objectives_list
                    )

                    iteration_spacing = spacing_metric(
                        current_pareto_front_objectives_iter,
                        maximize_objectives_list
                    )
                except Exception as e:
                    print(f"Error calculando métricas en iteración {iteration+1}: {e}")

        iter_execution_time = time.time() - iter_start_time

        # Almacenar métricas
        iteration_metrics['iteration'].append(iteration + 1)
        iteration_metrics['pareto_front_size'].append(iteration_pareto_size)
        iteration_metrics['hypervolume'].append(iteration_hv)
        iteration_metrics['arp'].append(iteration_arp)
        iteration_metrics['spacing'].append(iteration_spacing)
        iteration_metrics['execution_time'].append(iter_execution_time)

        # Imprimir resumen cada 10 iteraciones
        if (iteration + 1) % 10 == 0:
            print(f"  Iteración {iteration+1}: PF Size={iteration_pareto_size}, "
                  f"HV={iteration_hv:.4f}, ARP={iteration_arp:.4f}, "
                  f"Spacing={iteration_spacing:.4f}, Time={iter_execution_time:.2f}s")

    total_execution_time = time.time() - start_time_total

    print(f"\nCompletado MRL-AMIS para {grupo_info['id']} en {total_execution_time:.2f} segundos")

    # Extraer el frente de Pareto final
    final_population_objectives_feasible = np.array([
        [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
         obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
         obj_dict.get('riesgo_total', 0)]
        for sol in current_population_results
        if sol.get('is_feasible', False)
        for obj_dict in [sol.get('objetivos', {})]
        if obj_dict
    ])

    # MODIFICACIÓN IMPORTANTE: Calcular objetivos para población completa
    final_population_objectives_all = np.array([
        [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0), obj_dict.get('co2_total', 0),
         obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) + obj_dict.get('sust_social', 0),
         obj_dict.get('riesgo_total', 0)]
        for sol in current_population_results
        for obj_dict in [sol.get('objetivos', {})]
        if obj_dict
    ])

    final_pareto_front_objectives = []
    final_pareto_front_indices = []
    final_pareto_front_solutions = []

    if len(final_population_objectives_feasible) > 0:
        final_pareto_front_objectives, final_pareto_front_indices = find_pareto_front(
            final_population_objectives_feasible,
            maximize_objectives_list
        )

        final_feasible_solutions = [
            sol for sol in current_population_results
            if sol.get('is_feasible', False) and sol.get('objetivos') is not None and sol.get('objetivos')
        ]

        final_pareto_front_solutions = [final_feasible_solutions[i] for i in final_pareto_front_indices]

    # MODIFICACIÓN: Agregar cálculo de métricas finales incluyendo ARP
    final_metrics = {}
    if len(final_pareto_front_objectives) > 0:
        final_metrics['final_hypervolume'] = calculate_hypervolume(
            final_pareto_front_objectives,
            ref_point_hypervolume,
            maximize_objectives_list
        )
        final_metrics['final_arp'] = calculate_average_ratio_pareto(
            final_pareto_front_objectives,
            final_population_objectives_all,
            maximize_objectives_list
        )
        final_metrics['final_spacing'] = spacing_metric(
            final_pareto_front_objectives,
            maximize_objectives_list
        )

    return {
        'iteration_metrics': iteration_metrics,
        'final_pareto_front_objectives': final_pareto_front_objectives,
        'final_pareto_front_solutions': final_pareto_front_solutions,
        'execution_time': total_execution_time,
        'current_population_results': current_population_results,
        'final_metrics': final_metrics
    }


def run_mrl_amis_for_multiple_groups(data, groups_to_process, max_iterations_list=[100],
                                   num_work_packages=100, ref_point_hypervolume=None, maximize_objectives_list=None):
    """
    Ejecuta el algoritmo MRL-AMIS para múltiples grupos turísticos con diferentes números de iteraciones.
    """

    # Establecer valores por defecto a nivel global
    global_default_min_pois = 2
    global_default_max_pois = 10
    global_default_presupuesto = 500

    all_results = {}

    for grupo_id in groups_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESANDO {grupo_id.upper()}")
        print(f"{'='*60}")

        try:
            # Verificar si el grupo existe en los datos
            if grupo_id not in data["tourist_groups"].index:
                print(f"Error: El grupo {grupo_id} no existe en los datos.")
                continue

            # Obtener datos básicos del grupo con manejo de errores
            grupo_info = {"id": grupo_id}

            # Tiempo disponible
            if "tiempo_max_min" in data["tourist_groups"].columns:
                grupo_info["tiempo_disponible"] = data["tourist_groups"].loc[grupo_id, "tiempo_max_min"]
            else:
                grupo_info["tiempo_disponible"] = 480  # 8 horas como valor por defecto

            # Presupuesto - probar diferentes nombres de columna
            presupuesto_columnas = ["presupuesto_max_usd", "presupuesto_USD", "presupuesto", "budget"]
            presupuesto_encontrado = False
            for col in presupuesto_columnas:
                if col in data["tourist_groups"].columns:
                    presupuesto_val = data["tourist_groups"].loc[grupo_id, col]
                    if pd.notna(presupuesto_val):
                        grupo_info["presupuesto"] = presupuesto_val
                        presupuesto_encontrado = True
                        break

            if not presupuesto_encontrado:
                grupo_info["presupuesto"] = global_default_presupuesto
                print(f"Aviso: Usando presupuesto por defecto ({global_default_presupuesto}).")

            # Min POIs
            if "min_pois" in data["tourist_groups"].columns:
                min_pois_val = data["tourist_groups"].loc[grupo_id, "min_pois"]
                if pd.notna(min_pois_val):
                    grupo_info["min_pois_per_route"] = min_pois_val
                else:
                    grupo_info["min_pois_per_route"] = global_default_min_pois
            else:
                grupo_info["min_pois_per_route"] = global_default_min_pois

            # Max POIs
            if "max_pois" in data["tourist_groups"].columns:
                max_pois_val = data["tourist_groups"].loc[grupo_id, "max_pois"]
                if pd.notna(max_pois_val):
                    grupo_info["max_pois_per_route"] = max_pois_val
                else:
                    grupo_info["max_pois_per_route"] = global_default_max_pois
            else:
                grupo_info["max_pois_per_route"] = global_default_max_pois

            # Origen del grupo
            if "origen" in data["tourist_groups"].columns:
                origen_val = data["tourist_groups"].loc[grupo_id, "origen"]
                if pd.notna(origen_val):
                    grupo_info["origen"] = str(origen_val)
                else:
                    grupo_info["origen"] = '1'
            else:
                grupo_info["origen"] = '1'

            # Tipo de turista (necesario para cálculos)
            if "tipo_turista" in data["tourist_groups"].columns:
                tipo_turista = data["tourist_groups"].loc[grupo_id, "tipo_turista"]
            else:
                tipo_turista = "nacional"  # Valor por defecto

            print(f"Información del grupo:")
            print(f"  - Tipo turista: {tipo_turista}")
            print(f"  - Tiempo disponible: {grupo_info['tiempo_disponible']} minutos")
            print(f"  - POIs por ruta: {grupo_info['min_pois_per_route']} - {grupo_info['max_pois_per_route']}")
            print(f"  - Presupuesto: ${grupo_info['presupuesto']}")
            print(f"  - Origen: POI {grupo_info['origen']}")

            all_results[grupo_id] = {}

            for max_iterations in max_iterations_list:
                print(f"\n{'-'*40}")
                print(f"Ejecutando MRL-AMIS con {max_iterations} iteraciones")
                print(f"{'-'*40}")

                # Generar work packages iniciales
                wps_df = generar_work_packages(q=num_work_packages, d=len(data["pois"]))
                print(f"Generados {num_work_packages} Work Packages iniciales")

                # Decodificar y evaluar work packages iniciales
                resultados_wp_iniciales = []
                num_factibles = 0

                print("Decodificando y evaluando WPs iniciales...")
                for wp_name, wp_series in wps_df.iterrows():
                    wp_vector = wp_series.values

                    # Decodificar WP
                    ruta_decodificada, is_feasible = decodificar_wp(
                        wp_vector=wp_vector,
                        pois_df=data["pois"],
                        travel_times_df=data["travel_times"],
                        grupo_info=grupo_info,
                        origen=grupo_info['origen']
                    )

                    # Evaluar ruta
                    metrics = evaluar_funciones_objetivo(ruta_decodificada, data, tipo_turista=tipo_turista)

                    if is_feasible:
                        num_factibles += 1

                    resultados_wp_iniciales.append({
                        "wp_name": wp_name,
                        "wp_original": wp_vector,
                        "ruta_decodificada": ruta_decodificada,
                        "objetivos": metrics,
                        "is_feasible": is_feasible
                    })

                print(f"Soluciones factibles iniciales: {num_factibles}/{num_work_packages} ({num_factibles/num_work_packages*100:.1f}%)")

                # Ejecutar MRL-AMIS
                result = execute_mrl_amis(
                    resultados_wp_iniciales=resultados_wp_iniciales,
                    data=data,
                    grupo_info=grupo_info,
                    tipo_turista=tipo_turista,
                    max_iterations=max_iterations,
                    num_work_packages=num_work_packages,
                    maximize_objectives_list=maximize_objectives_list,
                    ref_point_hypervolume=ref_point_hypervolume
                )

                all_results[grupo_id][max_iterations] = result

                # Imprimir resumen de resultados para esta ejecución
                print(f"\nResultados para {grupo_id} con {max_iterations} iteraciones:")
                print(f"  - Tiempo total de ejecución: {result['execution_time']:.2f} segundos")
                print(f"  - Tamaño final del frente de Pareto: {len(result['final_pareto_front_solutions'])}")

                # Calcular métricas finales si hay frente de Pareto
                if len(result['final_pareto_front_objectives']) > 0:
                    final_hv = calculate_hypervolume(
                        result['final_pareto_front_objectives'],
                        ref_point_hypervolume,
                        maximize_objectives_list
                    )

                    # MODIFICACIÓN IMPORTANTE: Calcular ARP correctamente con población completa
                    final_arp = calculate_average_ratio_pareto(
                        result['final_pareto_front_objectives'],
                        np.array([
                            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0),
                             obj_dict.get('co2_total', 0),
                             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) +
                             obj_dict.get('sust_social', 0),
                             obj_dict.get('riesgo_total', 0)]
                            for sol in result['current_population_results']  # TODA LA POBLACIÓN
                            for obj_dict in [sol.get('objetivos', {})]      # NO solo factibles
                            if obj_dict                                     # Solo filtrar None
                        ]),
                        maximize_objectives_list
                    )

                    final_spacing = spacing_metric(
                        result['final_pareto_front_objectives'],
                        maximize_objectives_list
                    )

                    print(f"  - Hipervolumen final: {final_hv:.4f}")
                    print(f"  - ARP final: {final_arp:.4f}")
                    print(f"  - Spacing final: {final_spacing:.4f}")

                    # Imprimir mejor solución para cada objetivo
                    obj_names = ["Preferencia", "Costo", "CO2", "Sostenibilidad", "Riesgo"]
                    for i, name in enumerate(obj_names):
                        if maximize_objectives_list[i]:
                            best_idx = np.argmax(result['final_pareto_front_objectives'][:, i])
                        else:
                            best_idx = np.argmin(result['final_pareto_front_objectives'][:, i])

                        best_value = result['final_pareto_front_objectives'][best_idx, i]
                        print(f"  - Mejor {name}: {best_value:.2f}")

        except Exception as e:
            print(f"Error procesando {grupo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL DE RESULTADOS")
    print(f"{'='*60}")

    # Crear DataFrame para métricas
    metrics_data = []

    for grupo_id in groups_to_process:
        print(f"\n{grupo_id}:")
        for max_iter in max_iterations_list:
            if grupo_id in all_results and max_iter in all_results[grupo_id]:
                result = all_results[grupo_id][max_iter]
                print(f"  {max_iter} iteraciones:")
                print(f"    - Frente de Pareto: {len(result['final_pareto_front_solutions'])} soluciones")
                print(f"    - Tiempo de ejecución: {result['execution_time']:.2f} segundos")

                # Calcular hipervolumen
                final_hv = 0
                final_arp = 0
                if len(result['final_pareto_front_objectives']) > 0:
                    final_hv = calculate_hypervolume(
                        result['final_pareto_front_objectives'],
                        ref_point_hypervolume,
                        maximize_objectives_list
                    )
                    print(f"    - Hipervolumen: {final_hv:.4f}")

                    # MODIFICACIÓN IMPORTANTE: Calcular ARP correctamente en el resumen
                    final_arp = calculate_average_ratio_pareto(
                        result['final_pareto_front_objectives'],
                        np.array([
                            [obj_dict.get('preferencia_total', 0), obj_dict.get('costo_total', 0),
                             obj_dict.get('co2_total', 0),
                             obj_dict.get('sust_ambiental', 0) + obj_dict.get('sust_economica', 0) +
                             obj_dict.get('sust_social', 0),
                             obj_dict.get('riesgo_total', 0)]
                            for sol in result['current_population_results']  # TODA LA POBLACIÓN
                            for obj_dict in [sol.get('objetivos', {})]      # NO solo factibles
                            if obj_dict                                     # Solo filtrar None
                        ]),
                        maximize_objectives_list
                    )
                    print(f"    - ARP: {final_arp:.4f}")

                # Guardar métricas para exportar a Excel
                metrics_data.append({
                    "Grupo": grupo_id,
                    "Iteraciones": max_iter,
                    "Soluciones_Pareto": len(result['final_pareto_front_solutions']),
                    "Hipervolumen": final_hv,
                    "ARP": final_arp,  # AGREGAR ARP a la tabla
                    "Tiempo_Ejecucion": result['execution_time']
                })

    # Guardar métricas en Excel si hay datos
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        excel_path = "resultados_mrl_amis.xlsx"
        metrics_df.to_excel(excel_path, index=False)
        print(f"\nResultados guardados en: {excel_path}")

    return all_results