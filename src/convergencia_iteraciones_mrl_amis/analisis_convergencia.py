import numpy as np
from analisis_multi_objetivo_y_metricos.set_coverage import set_coverage
from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import calculate_average_ratio_pareto, calculate_hypervolume, inverted_generational_distance, spacing_metric
from ejecucion_iteraciones_mrl_amis.ejecutar_mrl_amis import execute_mrl_amis
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
from generacionWorkPackages.workPackages import decodificar_wp, generar_work_packages



def run_convergence_analysis(data, grupo_id, iterations_list,num_work_packages=100, ref_point_hypervolume=None, maximize_objectives_list=None):
    """
    Realiza un análisis de convergencia para un grupo turístico específico
    con diferentes números de iteraciones.

    Args:
        data: Datos del problema
        grupo_id: ID del grupo turístico
        iterations_list: Lista de números de iteraciones a probar

    Returns:
        dict: Resultados del análisis para cada número de iteraciones
    """
    convergence_results = {}

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE CONVERGENCIA PARA {grupo_id.upper()}")
    print(f"{'='*60}")

    # Obtener información del grupo
    grupo_info = {
        "id": grupo_id,
        "tiempo_disponible": data["tourist_groups"].loc[grupo_id, "tiempo_max_min"],
        'min_pois_per_route': 2,  # Valor por defecto
        'max_pois_per_route': 10  # Valor por defecto
    }

    # Agregar origen
    if "origen" in data["tourist_groups"].columns:
        grupo_info["origen"] = str(data["tourist_groups"].loc[grupo_id, "origen"])
    else:
        grupo_info["origen"] = "1"

    # Agregar presupuesto
    presupuesto_columnas = ["presupuesto_max_usd", "presupuesto_USD", "presupuesto", "budget"]
    for col in presupuesto_columnas:
        if col in data["tourist_groups"].columns:
            grupo_info["presupuesto"] = data["tourist_groups"].loc[grupo_id, col]
            break
    else:
        grupo_info["presupuesto"] = 500  # Valor por defecto

    # Obtener tipo de turista
    tipo_turista = data["tourist_groups"].loc[grupo_id, "tipo_turista"]

    print(f"Información del grupo:")
    print(f"  - Tipo turista: {tipo_turista}")
    print(f"  - Tiempo disponible: {grupo_info['tiempo_disponible']} minutos")
    print(f"  - POIs por ruta: {grupo_info['min_pois_per_route']} - {grupo_info['max_pois_per_route']}")
    print(f"  - Presupuesto: ${grupo_info['presupuesto']}")
    print(f"  - Origen: POI {grupo_info['origen']}")

    # Generar y evaluar soluciones iniciales (se hace una sola vez)
    print(f"\nGenerando soluciones iniciales para todos los análisis...")
    wps_df = generar_work_packages(q=num_work_packages, d=len(data["pois"]))

    resultados_wp_iniciales = []
    num_factibles = 0

    for wp_name, wp_series in wps_df.iterrows():
        wp_vector = wp_series.values

        ruta_decodificada, is_feasible = decodificar_wp(
            wp_vector=wp_vector,
            pois_df=data["pois"],
            travel_times_df=data["travel_times"],
            grupo_info=grupo_info,
            origen=grupo_info['origen']
        )

        metrics = evaluar_funciones_objetivo(ruta_decodificada, data, tipo_turista)

        if is_feasible:
            num_factibles += 1

        resultados_wp_iniciales.append({
            "wp_name": wp_name,
            "wp_original": wp_vector,
            "ruta_decodificada": ruta_decodificada,
            "objetivos": metrics,
            "is_feasible": is_feasible
        })

    print(f"Soluciones factibles iniciales: {num_factibles}/{len(resultados_wp_iniciales)} ({num_factibles/len(resultados_wp_iniciales)*100:.1f}%)")

    # Ejecutar con diferentes números de iteraciones
    for max_iterations in iterations_list:
        print(f"\n{'-'*40}")
        print(f"Ejecutando con {max_iterations} iteraciones")
        print(f"{'-'*40}")

        # Usar las mismas soluciones iniciales para todas las ejecuciones
        result = execute_mrl_amis(
            resultados_wp_iniciales=resultados_wp_iniciales.copy(),
            data=data,
            grupo_info=grupo_info,
            tipo_turista=tipo_turista,
            max_iterations=max_iterations,
            num_work_packages=num_work_packages
        )

        convergence_results[max_iterations] = result

        # Imprimir resultados
        print(f"\nResultados con {max_iterations} iteraciones:")
        print(f"  - Tiempo de ejecución: {result['execution_time']:.2f} segundos")
        print(f"  - Tamaño del frente de Pareto: {len(result['final_pareto_front_solutions'])}")

        # Calcular todas las métricas
        if len(result['final_pareto_front_objectives']) > 0:
            # Hipervolumen
            final_hv = calculate_hypervolume(
                result['final_pareto_front_objectives'],
                ref_point_hypervolume,
                maximize_objectives_list
            )
            print(f"  - Hipervolumen: {final_hv:.4f}")

            # ARP
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

            final_arp = calculate_average_ratio_pareto(
                result['final_pareto_front_objectives'],
                all_objectives,
                maximize_objectives_list
            )
            print(f"  - ARP: {final_arp:.4f}")

            # Spacing
            final_spacing = spacing_metric(
                result['final_pareto_front_objectives'],
                maximize_objectives_list
            )
            print(f"  - Spacing: {final_spacing:.4f}")

            # IGD (si hay frente de referencia)
            if max_iterations == max(iterations_list):
                # Usar el mejor frente encontrado como referencia
                result['reference_front'] = result['final_pareto_front_objectives']
            elif 'reference_front' in convergence_results.get(1000, {}):
                final_igd = inverted_generational_distance(
                    result['final_pareto_front_objectives'],
                    convergence_results[1000]['reference_front'],
                    maximize_objectives_list
                )
                print(f"  - IGD: {final_igd:.4f}")

    # Generar visualizaciones de la convergencia
    metrics_df = plot_convergence_metrics(convergence_results, grupo_id, iterations_list, ref_point_hypervolume, maximize_objectives_list)

    # Calcular Set Coverage entre diferentes iteraciones
    calculate_set_coverage_analysis(convergence_results, grupo_id, iterations_list, maximize_objectives_list)

    return convergence_results

def plot_convergence_metrics(convergence_results, grupo_id, iterations_list, ref_point_hypervolume=None, maximize_objectives_list=None):
    """
    Genera gráficos para visualizar la convergencia de las métricas.
    """
    import matplotlib.pyplot as plt

    # Extraer métricas
    hypervolume_values = []
    pf_size_values = []
    spacing_values = []
    arp_values = []
    igd_values = []
    time_values = []

    for iters in iterations_list:
        result = convergence_results[iters]

        # Hipervolumen
        if len(result['final_pareto_front_objectives']) > 0:
            hv = calculate_hypervolume(
                result['final_pareto_front_objectives'],
                ref_point_hypervolume,
                maximize_objectives_list, 
            )
        else:
            hv = 0
        hypervolume_values.append(hv)

        # Tamaño del frente de Pareto
        pf_size_values.append(len(result['final_pareto_front_solutions']))

        # Spacing
        if len(result['final_pareto_front_objectives']) > 0:
            sp = spacing_metric(
                result['final_pareto_front_objectives'],
                maximize_objectives_list
            )
        else:
            sp = 0
        spacing_values.append(sp)

        # ARP
        if len(result['final_pareto_front_objectives']) > 0:
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
        else:
            arp = 0
        arp_values.append(arp)

        # IGD
        if 'reference_front' in convergence_results.get(max(iterations_list), {}):
            if len(result['final_pareto_front_objectives']) > 0:
                igd = inverted_generational_distance(
                    result['final_pareto_front_objectives'],
                    convergence_results[max(iterations_list)]['reference_front'],
                    maximize_objectives_list
                )
            else:
                igd = float('inf')
        else:
            igd = None
        igd_values.append(igd)

        # Tiempo
        time_values.append(result['execution_time'])

    # Crear gráficos con 6 subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

    # Gráfico de hipervolumen
    ax1.plot(iterations_list, hypervolume_values, 'o-', color='blue')
    ax1.set_xlabel('Número de iteraciones')
    ax1.set_ylabel('Hipervolumen')
    ax1.set_title(f'Convergencia de Hipervolumen - {grupo_id}')
    ax1.grid(True)

    # Gráfico de tamaño del frente de Pareto
    ax2.plot(iterations_list, pf_size_values, 'o-', color='green')
    ax2.set_xlabel('Número de iteraciones')
    ax2.set_ylabel('Tamaño del frente de Pareto')
    ax2.set_title(f'Tamaño del frente de Pareto - {grupo_id}')
    ax2.grid(True)

    # Gráfico de spacing
    ax3.plot(iterations_list, spacing_values, 'o-', color='red')
    ax3.set_xlabel('Número de iteraciones')
    ax3.set_ylabel('Spacing')
    ax3.set_title(f'Spacing - {grupo_id}')
    ax3.grid(True)

    # Gráfico de ARP
    ax4.plot(iterations_list, arp_values, 'o-', color='orange')
    ax4.set_xlabel('Número de iteraciones')
    ax4.set_ylabel('ARP')
    ax4.set_title(f'Average Ratio of Pareto Solutions - {grupo_id}')
    ax4.grid(True)

    # Gráfico de IGD
    ax5.plot(iterations_list, [v for v in igd_values if v is not None], 'o-', color='purple')
    ax5.set_xlabel('Número de iteraciones')
    ax5.set_ylabel('IGD')
    ax5.set_title(f'Inverted Generational Distance - {grupo_id}')
    ax5.grid(True)

    # Gráfico de tiempo de ejecución
    ax6.plot(iterations_list, time_values, 'o-', color='brown')
    ax6.set_xlabel('Número de iteraciones')
    ax6.set_ylabel('Tiempo de ejecución (s)')
    ax6.set_title(f'Tiempo de ejecución - {grupo_id}')
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(f'convergencia_completa_{grupo_id}.png', dpi=300)
    plt.close()

    print(f"Gráficos de convergencia guardados como 'convergencia_completa_{grupo_id}.png'")

    # Crear tabla con todas las métricas
    import pandas as pd

    metrics_df = pd.DataFrame({
        'Iteraciones': iterations_list,
        'Hipervolumen': hypervolume_values,
        'Tamaño_PF': pf_size_values,
        'Spacing': spacing_values,
        'ARP': arp_values,
        'IGD': [v if v is not None else 'N/A' for v in igd_values],
        'Tiempo_Ejecución_s': time_values
    })

    # Agregar porcentajes de convergencia
    metrics_df['Conv_HV_%'] = (metrics_df['Hipervolumen'] / metrics_df['Hipervolumen'].max() * 100).round(2)
    metrics_df['Conv_ARP_%'] = (metrics_df['ARP'] / metrics_df['ARP'].max() * 100).round(2)

    metrics_df.to_excel(f'metricas_convergencia_completa_{grupo_id}.xlsx', index=False)
    print(f"Tabla de métricas guardada como 'metricas_convergencia_completa_{grupo_id}.xlsx'")

    # Imprimir tabla resumen
    print(f"\nRESUMEN DE MÉTRICAS PARA {grupo_id}:")
    print(metrics_df.round(4))

    return metrics_df

def calculate_set_coverage_analysis(convergence_results, grupo_id, iterations_list, maximize_objectives_list=None):
    """
    Calcula el Set Coverage entre diferentes números de iteraciones.
    """
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE SET COVERAGE - {grupo_id.upper()}")
    print(f"{'='*60}")

    # Crear matriz de set coverage
    coverage_matrix = np.zeros((len(iterations_list), len(iterations_list)))

    for i, iter_i in enumerate(iterations_list):
        for j, iter_j in enumerate(iterations_list):
            if i != j:
                result_i = convergence_results[iter_i]
                result_j = convergence_results[iter_j]

                if len(result_i['final_pareto_front_objectives']) > 0 and len(result_j['final_pareto_front_objectives']) > 0:
                    coverage = set_coverage(
                        result_i['final_pareto_front_objectives'],
                        result_j['final_pareto_front_objectives'],
                        maximize_objectives_list
                    )
                    coverage_matrix[i, j] = coverage

    # Crear DataFrame para visualización
    coverage_df = pd.DataFrame(
        coverage_matrix,
        index=[f'{iters} iter' for iters in iterations_list],
        columns=[f'{iters} iter' for iters in iterations_list]
    )

    print("\nMatriz de Set Coverage:")
    print("(Fila X cubre a Columna Y)")
    print(coverage_df.round(4))

    # Guardar en Excel
    coverage_df.to_excel(f'set_coverage_{grupo_id}.xlsx')
    print(f"\nMatriz de Set Coverage guardada en 'set_coverage_{grupo_id}.xlsx'")

    # Visualización
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(coverage_df, annot=True, fmt='.4f', cmap='YlOrRd', vmin=0, vmax=1)
    plt.title(f'Set Coverage Analysis - {grupo_id}')
    plt.tight_layout()
    plt.savefig(f'set_coverage_heatmap_{grupo_id}.png', dpi=300)
    plt.close()

    print(f"Mapa de calor de Set Coverage guardado como 'set_coverage_heatmap_{grupo_id}.png'")

    # Análisis de tendencias
    print("\nAnálisis de tendencias:")
    for i, iter_i in enumerate(iterations_list[:-1]):
        next_coverage = coverage_matrix[i+1, i]
        print(f"  - {iterations_list[i+1]} iteraciones cubre {next_coverage:.4f} de {iter_i} iteraciones")

    return coverage_matrix

