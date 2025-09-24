import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ejecucion_iteraciones_mrl_amis.ejecutar_mrl_amis import execute_mrl_amis
import folium
from folium.plugins import PolyLineTextPath, MarkerCluster
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo
from generacionWorkPackages.workPackages import decodificar_wp, generar_work_packages

def identify_optimal_routes_topsis(data, groups_to_process, iterations_for_analysis=1000, num_work_packages=100, maximize_objectives_list=[True, False, False, True, False]):
    """
    Identifica las rutas óptimas para cada grupo turístico usando TOPSIS.

    Args:
        data: Datos del problema
        groups_to_process: Lista de IDs de grupos para analizar
        iterations_for_analysis: Número de iteraciones a usar para obtener el frente de Pareto

    Returns:
        dict: Rutas óptimas para cada grupo
    """
    optimal_routes = {}

    print(f"\n{'='*60}")
    print(f"IDENTIFICACIÓN DE RUTAS ÓPTIMAS CON TOPSIS")
    print(f"{'='*60}")

    for grupo_id in groups_to_process:
        print(f"\n{'-'*40}")
        print(f"Analizando {grupo_id.upper()}")
        print(f"{'-'*40}")

        try:
            # Verificar si el grupo existe en los datos
            if grupo_id not in data["tourist_groups"].index:
                print(f"Error: El grupo {grupo_id} no existe en los datos.")
                continue

            # Obtener información del grupo
            grupo_info = {"id": grupo_id}

            # Tiempo disponible
            if "tiempo_max_min" in data["tourist_groups"].columns:
                grupo_info["tiempo_disponible"] = data["tourist_groups"].loc[grupo_id, "tiempo_max_min"]
            else:
                grupo_info["tiempo_disponible"] = 480  # 8 horas como valor por defecto

            # Agregar otros parámetros necesarios
            grupo_info["origen"] = str(data["tourist_groups"].loc[grupo_id, "origen"]) if "origen" in data["tourist_groups"].columns else "1"
            grupo_info["min_pois_per_route"] = 2
            grupo_info["max_pois_per_route"] = 10

            # Obtener tipo de turista
            tipo_turista = data["tourist_groups"].loc[grupo_id, "tipo_turista"] if "tipo_turista" in data["tourist_groups"].columns else "nacional"

            print(f"Grupo: {grupo_id}, Tipo: {tipo_turista}, Origen: {grupo_info['origen']}")

            # Ejecutar MRL-AMIS con el número de iteraciones especificado
            print(f"Ejecutando MRL-AMIS con {iterations_for_analysis} iteraciones...")

            # Generar población inicial
            wps_df = generar_work_packages(q=num_work_packages, d=len(data["pois"]))

            resultados_wp_iniciales = []
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

                resultados_wp_iniciales.append({
                    "wp_name": wp_name,
                    "wp_original": wp_vector,
                    "ruta_decodificada": ruta_decodificada,
                    "objetivos": metrics,
                    "is_feasible": is_feasible
                })

            # Ejecutar algoritmo
            result = execute_mrl_amis(
                resultados_wp_iniciales=resultados_wp_iniciales,
                data=data,
                grupo_info=grupo_info,
                tipo_turista=tipo_turista,
                max_iterations=iterations_for_analysis,
                num_work_packages=num_work_packages
            )

            # Verificar si hay soluciones en el frente de Pareto
            if len(result['final_pareto_front_solutions']) == 0:
                print(f"No se encontraron soluciones para {grupo_id}.")
                continue

            print(f"Encontradas {len(result['final_pareto_front_solutions'])} soluciones en el frente de Pareto.")

            # Aplicar TOPSIS para identificar la mejor ruta
            best_routes = apply_topsis(result['final_pareto_front_solutions'],
                                      result['final_pareto_front_objectives'],
                                      maximize_objectives_list)

            # Generar visualizaciones
            visualize_topsis_results(best_routes, grupo_id, data, tipo_turista)

            # Visualizar mapa para la mejor ruta
            create_route_map(best_routes[0]['solution']['ruta_decodificada'],
                            data,
                            f"{grupo_id}_mejor_ruta")

            optimal_routes[grupo_id] = best_routes

        except Exception as e:
            print(f"Error procesando {grupo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generar informe comparativo
    generate_comparative_report(optimal_routes, data)

    return optimal_routes

def apply_topsis(pareto_solutions, pareto_objectives, maximize_objectives_list, num_top_solutions=5):
    """
    Aplica el método TOPSIS para clasificar soluciones del frente de Pareto.

    Args:
        pareto_solutions: Lista de soluciones en el frente de Pareto
        pareto_objectives: Matriz de objetivos de las soluciones
        maximize_objectives_list: Lista de objetivos a maximizar
        num_top_solutions: Número de mejores soluciones a retornar

    Returns:
        list: Lista de mejores soluciones según TOPSIS
    """
    # 1. Normalizar la matriz de decisión
    norm = np.linalg.norm(pareto_objectives, axis=0)
    normalized_matrix = pareto_objectives / norm

    # 2. Definir pesos para los objetivos (asignar iguales por defecto)
    weights = np.array([0.25, 0.20, 0.15, 0.25, 0.15])  # Preferencia, Costo, CO2, Sostenibilidad, Riesgo

    # 3. Calcular matriz de decisión normalizada ponderada
    weighted_normalized = normalized_matrix * weights

    # 4. Determinar solución ideal y anti-ideal
    ideal_best = np.zeros(pareto_objectives.shape[1])
    ideal_worst = np.zeros(pareto_objectives.shape[1])

    for i, maximize in enumerate(maximize_objectives_list):
        if maximize:
            ideal_best[i] = np.max(weighted_normalized[:, i])
            ideal_worst[i] = np.min(weighted_normalized[:, i])
        else:
            ideal_best[i] = np.min(weighted_normalized[:, i])
            ideal_worst[i] = np.max(weighted_normalized[:, i])

    # 5. Calcular distancias a ideal y anti-ideal
    dist_ideal = np.sqrt(np.sum((weighted_normalized - ideal_best)**2, axis=1))
    dist_anti_ideal = np.sqrt(np.sum((weighted_normalized - ideal_worst)**2, axis=1))

    # 6. Calcular proximidad relativa
    proximity = dist_anti_ideal / (dist_ideal + dist_anti_ideal)

    # 7. Clasificar soluciones
    ranking_indices = np.argsort(proximity)[::-1]  # De mayor a menor proximidad

    # 8. Seleccionar top soluciones
    top_solutions = []
    for i in range(min(num_top_solutions, len(ranking_indices))):
        idx = ranking_indices[i]
        top_solutions.append({
            'rank': i+1,
            'solution': pareto_solutions[idx],
            'objectives': pareto_objectives[idx],
            'topsis_score': proximity[idx]
        })

    return top_solutions

def visualize_topsis_results(best_routes, grupo_id, data, tipo_turista, maximize_objectives_list=[True, False, False, True, False]):
    """
    Visualiza los resultados del análisis TOPSIS.

    Args:
        best_routes: Lista de mejores rutas según TOPSIS
        grupo_id: ID del grupo turístico
        data: Datos del problema
        tipo_turista: Tipo de turista
    """
    print(f"\n{'-'*40}")
    print(f"RESULTADOS TOPSIS PARA {grupo_id.upper()}")
    print(f"{'-'*40}")

    # Crear dataframe con resultados
    results_data = []

    obj_names = ["Preferencia", "Costo", "CO2", "Sostenibilidad", "Riesgo"]
    for route in best_routes:
        route_data = {
            'Ranking': route['rank'],
            'TOPSIS Score': route['topsis_score'],
            'Num POIs': len(route['solution']['ruta_decodificada']) - 2  # -2 por origen duplicado
        }

        # Agregar objetivos
        for i, name in enumerate(obj_names):
            route_data[name] = route['objectives'][i]

        # Agregar POIs visitados
        route_data['POIs Visitados'] = ' → '.join(route['solution']['ruta_decodificada'])

        results_data.append(route_data)

    results_df = pd.DataFrame(results_data)

    # Guardar en Excel
    results_df.to_excel(f'topsis_results_{grupo_id}.xlsx', index=False)
    print(f"Resultados guardados en 'topsis_results_{grupo_id}.xlsx'")

    # Imprimir resultados en consola
    print("\nMejores rutas según TOPSIS:")
    for route in best_routes:
        print(f"\nRanking #{route['rank']} (Score: {route['topsis_score']:.4f})")
        print(f"  Ruta: {' → '.join(route['solution']['ruta_decodificada'])}")
        print(f"  Objetivos:")
        for i, name in enumerate(obj_names):
            value = route['objectives'][i]
            print(f"   - {name}: {value:.2f}")

    # Generar visualización de radar chart para las 3 mejores rutas
    create_radar_chart(best_routes[:min(3, len(best_routes))], obj_names, maximize_objectives_list, grupo_id)

def create_radar_chart(top_routes, categories, maximize_objectives_list, grupo_id):
    """
    Crea un gráfico de radar para comparar las mejores rutas.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Preparar datos para el radar chart
    n_cats = len(categories)
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Normalizar objetivos entre 0-1
    all_objectives = np.array([route['objectives'] for route in top_routes])
    min_vals = np.min(all_objectives, axis=0)
    max_vals = np.max(all_objectives, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Evitar división por cero

    for i, route in enumerate(top_routes):
        values = route['objectives'].copy()

        # Invertir valores para objetivos a minimizar
        for j, maximize in enumerate(maximize_objectives_list):
            if not maximize:
                values[j] = max_vals[j] - values[j] + min_vals[j]

        # Normalizar valores entre 0 y 1
        norm_values = (values - min_vals) / range_vals

        # Cerrar el polígono
        norm_values = np.append(norm_values, norm_values[0])

        # Plotear
        ax.plot(angles, norm_values, linewidth=2, linestyle='solid',
                label=f"Ruta #{route['rank']}")
        ax.fill(angles, norm_values, alpha=0.1)

    # Configurar el gráfico
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Comparación de las mejores rutas - {grupo_id}', size=15)

    plt.tight_layout()
    plt.savefig(f'radar_chart_{grupo_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gráfico de radar guardado como 'radar_chart_{grupo_id}.png'")

def create_route_map(route, data, map_name):
    """
    Crea un mapa interactivo con la ruta.

    Args:
        route: Lista de POIs en la ruta
        data: Datos del problema
        map_name: Nombre del archivo del mapa
    """
    # Verificar si tenemos coordenadas para los POIs
    if 'latitud' not in data['pois'].columns or 'longitud' not in data['pois'].columns:
        print("No se pueden crear mapas: faltan coordenadas en los datos de POIs")
        return

    # Obtener coordenadas para los POIs de la ruta
    coords = []
    poi_names = []

    for poi in route:
        if poi in data['pois'].index:
            lat = data['pois'].loc[poi, 'latitud']
            lon = data['pois'].loc[poi, 'longitud']
            name = data['pois'].loc[poi, 'nombre'] if 'nombre' in data['pois'].columns else f"POI {poi}"

            coords.append([lat, lon])
            poi_names.append(name)

    # Verificar si tenemos coordenadas válidas
    if not coords or len(coords) < 2:
        print(f"No se puede crear mapa para la ruta: faltan coordenadas")
        return

    # Crear mapa centrado en el primer POI
    center_lat = sum(c[0] for c in coords) / len(coords)
    center_lon = sum(c[1] for c in coords) / len(coords)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Agregar marcadores para cada POI
    for i, (coord, name) in enumerate(zip(coords, poi_names)):
        # Estilo especial para origen/destino
        if i == 0 or i == len(coords) - 1:
            icon = folium.Icon(color='red', icon='home')
            popup = f"Origen/Destino: {name}"
        else:
            icon = folium.Icon(color='blue', icon='info-sign')
            popup = f"POI #{i}: {name}"

        folium.Marker(
            location=coord,
            popup=popup,
            tooltip=name,
            icon=icon
        ).add_to(m)

    # Agregar línea de ruta
    folium.PolyLine(
        locations=coords,
        color='green',
        weight=4,
        opacity=0.7,
        tooltip="Ruta"
    ).add_to(m)

    # Guardar mapa
    m.save(f'{map_name}.html')
    print(f"Mapa guardado como '{map_name}.html'")

def generate_comparative_report(optimal_routes, data):
    """
    Genera un informe comparativo de las mejores rutas para cada grupo.
    """
    if not optimal_routes:
        print("No hay rutas para generar informe comparativo")
        return

    print(f"\n{'='*60}")
    print(f"INFORME COMPARATIVO DE RUTAS ÓPTIMAS")
    print(f"{'='*60}")

    # Preparar datos para el informe
    report_data = []

    for grupo_id, routes in optimal_routes.items():
        if not routes:
            continue

        best_route = routes[0]

        report_row = {
            'Grupo': grupo_id,
            'TOPSIS Score': best_route['topsis_score'],
            'Num POIs': len(best_route['solution']['ruta_decodificada']) - 2,
            'Preferencia': best_route['objectives'][0],
            'Costo': best_route['objectives'][1],
            'CO2': best_route['objectives'][2],
            'Sostenibilidad': best_route['objectives'][3],
            'Riesgo': best_route['objectives'][4],
            'Ruta': ' → '.join(best_route['solution']['ruta_decodificada'])
        }

        report_data.append(report_row)

    # Crear DataFrame
    report_df = pd.DataFrame(report_data)

    # Guardar en Excel
    report_df.to_excel('informe_comparativo_rutas.xlsx', index=False)
    print("Informe comparativo guardado en 'informe_comparativo_rutas.xlsx'")

    # Visualización comparativa
    create_comparative_charts(report_df)

    return report_df

def create_comparative_charts(report_df):
    """
    Crea gráficos comparativos de las mejores rutas para cada grupo.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Configuración general
    plt.style.use('ggplot')

    # 1. Comparativa de objetivos por grupo
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    metrics = ['Preferencia', 'Costo', 'CO2', 'Sostenibilidad', 'Riesgo', 'Num POIs']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        sns.barplot(x='Grupo', y=metric, data=report_df, ax=axs[i], color=color)
        axs[i].set_title(f'Comparativa de {metric}')
        axs[i].set_ylabel(metric)
        axs[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('comparativa_objetivos.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Gráfico de radar
    create_comparative_radar(report_df)

    print("Gráficos comparativos guardados como 'comparativa_objetivos.png' y 'comparativa_radar.png'")

def create_comparative_radar(report_df):
    """
    Crea un gráfico de radar comparativo para todos los grupos.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Preparar datos
    categories = ['Preferencia', 'Costo', 'CO2', 'Sostenibilidad', 'Riesgo']
    maximize = [True, False, False, True, False]  # Corresponde a maximize_objectives_list

    groups = report_df['Grupo'].tolist()
    n_groups = len(groups)

    # Obtener valores de objetivos
    values = report_df[categories].values

    # Normalizar valores entre 0-1
    min_vals = np.min(values, axis=0)
    max_vals = np.max(values, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Evitar división por cero

    # Invertir escala para objetivos a minimizar
    for i, max_obj in enumerate(maximize):
        if not max_obj:
            values[:, i] = max_vals[i] - values[:, i] + min_vals[i]

    # Normalizar
    values_norm = (values - min_vals) / range_vals

    # Configurar gráfico
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Colores para cada grupo
    colors = plt.cm.viridis(np.linspace(0, 1, n_groups))

    # Plotear cada grupo
    for i, group in enumerate(groups):
        values_group = values_norm[i, :].tolist()
        values_group += values_group[:1]  # Cerrar el polígono

        ax.plot(angles, values_group, linewidth=2, linestyle='solid',
                label=group, color=colors[i])
        ax.fill(angles, values_group, alpha=0.1, color=colors[i])

    # Configurar el gráfico
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comparación de rutas óptimas entre grupos', size=15)

    plt.tight_layout()
    plt.savefig('comparativa_radar.png', dpi=300, bbox_inches='tight')
    plt.close()