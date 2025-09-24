def create_realistic_route_maps(optimal_routes, data, selected_groups=['grupo_2', 'grupo_3'], top_n=3):
    """
    Crea mapas con rutas realistas siguiendo carreteras para los grupos seleccionados.

    Args:
        optimal_routes: Diccionario con las rutas óptimas generadas por TOPSIS
        data: Datos del problema
        selected_groups: Lista de grupos para generar mapas
        top_n: Número de mejores rutas a visualizar
    """
    import folium
    from folium.plugins import MarkerCluster
    import requests
    import polyline
    import time

    print(f"\n{'='*60}")
    print(f"GENERANDO MAPAS CON RUTAS REALISTAS")
    print(f"{'='*60}")

    # Verificar coordenadas en datos
    if 'latitud' not in data['pois'].columns or 'longitud' not in data['pois'].columns:
        print("No se pueden crear mapas: faltan coordenadas en los datos de POIs")
        return

    for grupo_id in selected_groups:
        if grupo_id not in optimal_routes:
            print(f"No hay datos para {grupo_id}")
            continue

        print(f"\nGenerando mapas para {grupo_id.upper()}...")

        # Seleccionar top N rutas
        routes_to_map = optimal_routes[grupo_id][:min(top_n, len(optimal_routes[grupo_id]))]

        for route_idx, route_data in enumerate(routes_to_map):
            rank = route_data['rank']
            score = route_data['topsis_score']
            route = route_data['solution']['ruta_decodificada']
            objectives = route_data['objectives']

            print(f"  Procesando ruta #{rank} (Score: {score:.4f})...")

            # Obtener coordenadas para los POIs de la ruta
            coords = []
            poi_names = []

            for poi in route:
                if poi in data['pois'].index:
                    lat = data['pois'].loc[poi, 'latitud']
                    lon = data['pois'].loc[poi, 'longitud']
                    name = data['pois'].loc[poi, 'nombre'] if 'nombre' in data['pois'].columns else f"POI {poi}"

                    coords.append((lat, lon))
                    poi_names.append(name)

            # Verificar si tenemos coordenadas válidas
            if not coords or len(coords) < 2:
                print(f"    No se puede crear mapa: faltan coordenadas")
                continue

            # Crear mapa centrado en el recorrido
            center_lat = sum(c[0] for c in coords) / len(coords)
            center_lon = sum(c[1] for c in coords) / len(coords)

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Agregar marcadores para cada POI
            marker_cluster = MarkerCluster().add_to(m)

            for i, ((lat, lon), name) in enumerate(zip(coords, poi_names)):
                # Estilo especial para origen/destino
                if i == 0 or i == len(coords) - 1:
                    icon = folium.Icon(color='red', icon='home')
                    popup = f"<b>Origen/Destino:</b> {name}"
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        tooltip=name,
                        icon=icon
                    ).add_to(m)
                else:
                    icon = folium.Icon(color='blue', icon='info-sign')
                    popup = f"<b>POI #{i}:</b> {name}"
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup,
                        tooltip=f"{i}. {name}",
                        icon=icon
                    ).add_to(marker_cluster)

            # Obtener y agregar rutas realistas entre POIs
            for i in range(len(coords) - 1):
                # Obtener coordenadas de origen y destino
                origin = coords[i]
                destination = coords[i + 1]

                # Usar OSRM (OpenStreetMap Routing Machine) para obtener la ruta
                try:
                    # Formato: lon,lat (OSRM usa longitud primero)
                    url = f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=full&geometries=polyline"
                    response = requests.get(url)

                    if response.status_code == 200:
                        route_data = response.json()

                        if 'routes' in route_data and len(route_data['routes']) > 0:
                            # Decodificar la polyline
                            route_coords = polyline.decode(route_data['routes'][0]['geometry'])

                            # Agregar la ruta al mapa
                            folium.PolyLine(
                                locations=route_coords,
                                color='green' if i == 0 else ('blue' if i == len(coords) - 2 else 'purple'),
                                weight=4,
                                opacity=0.8,
                                tooltip=f"Ruta {poi_names[i]} → {poi_names[i+1]}"
                            ).add_to(m)

                            print(f"    Segmento {i+1}: {poi_names[i]} → {poi_names[i+1]} añadido")
                        else:
                            print(f"    No se encontró ruta para el segmento {i+1}")
                    else:
                        print(f"    Error en la API de rutas para el segmento {i+1}: {response.status_code}")

                    # Esperar un poco para no sobrecargar la API
                    time.sleep(0.5)

                except Exception as e:
                    print(f"    Error obteniendo ruta para segmento {i+1}: {e}")
                    # Crear línea directa como fallback
                    folium.PolyLine(
                        locations=[[origin[0], origin[1]], [destination[0], destination[1]]],
                        color='red',
                        weight=2,
                        opacity=0.5,
                        dash_array='5',
                        tooltip=f"Conexión directa {poi_names[i]} → {poi_names[i+1]}",
                        popup="Ruta estimada (no disponible ruta real)"
                    ).add_to(m)

            # Añadir leyenda al mapa
            legend_html = '''
             <div style="position: fixed;
                         bottom: 50px; right: 50px; width: 250px; height: 190px;
                         border:2px solid grey; z-index:9999; font-size:14px;
                         background-color:white; padding: 10px;
                         border-radius: 6px;">
             <div style="text-align: center; font-weight: bold; margin-bottom: 10px;">Ruta #{} - TOPSIS Score: {:.4f}</div>
             <div style="margin-top: 5px;"><b>Objetivos:</b></div>
             <div style="margin-left: 10px;">• Preferencia: {:.2f}</div>
             <div style="margin-left: 10px;">• Costo: {:.2f}</div>
             <div style="margin-left: 10px;">• CO2: {:.2f}</div>
             <div style="margin-left: 10px;">• Sostenibilidad: {:.2f}</div>
             <div style="margin-left: 10px;">• Riesgo: {:.2f}</div>
             <div style="margin-top: 5px;"><b>POIs:</b> {}</div>
             </div>
             '''.format(rank, score,
                        objectives[0], objectives[1], objectives[2], objectives[3], objectives[4],
                        len(coords) - 2)

            m.get_root().html.add_child(folium.Element(legend_html))

            # Guardar mapa
            map_file = f'{grupo_id}_ruta_{rank}_real.html'
            m.save(map_file)
            print(f"    Mapa guardado como '{map_file}'")

        # Crear un mapa combinado con las top N rutas
        print(f"\n  Generando mapa combinado para las {top_n} mejores rutas de {grupo_id}...")
        create_combined_map(routes_to_map, data, grupo_id, top_n)

    print("\nTodos los mapas fueron generados exitosamente.")

def create_combined_map(routes, data, grupo_id, top_n):
    """
    Crea un mapa que muestra las top N rutas en diferentes colores.
    """
    import folium
    from folium.plugins import MarkerCluster
    import requests
    import polyline
    import time

    # Obtener coordenadas de todas las rutas
    all_coords = []

    for route_data in routes:
        route = route_data['solution']['ruta_decodificada']
        for poi in route:
            if poi in data['pois'].index:
                lat = data['pois'].loc[poi, 'latitud']
                lon = data['pois'].loc[poi, 'longitud']
                all_coords.append((lat, lon))

    # Verificar si tenemos coordenadas válidas
    if not all_coords:
        print(f"  No se puede crear mapa combinado: faltan coordenadas")
        return

    # Crear mapa centrado en la media de todas las coordenadas
    center_lat = sum(c[0] for c in all_coords) / len(all_coords)
    center_lon = sum(c[1] for c in all_coords) / len(all_coords)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # Definir colores para cada ruta
    route_colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Agregar cada ruta en un color diferente
    for route_idx, route_data in enumerate(routes):
        rank = route_data['rank']
        route = route_data['solution']['ruta_decodificada']
        color = route_colors[route_idx % len(route_colors)]

        # Crear capa de ruta
        route_layer = folium.FeatureGroup(name=f"Ruta #{rank}", show=True)

        # Obtener coordenadas para los POIs de la ruta
        coords = []
        poi_names = []

        for poi in route:
            if poi in data['pois'].index:
                lat = data['pois'].loc[poi, 'latitud']
                lon = data['pois'].loc[poi, 'longitud']
                name = data['pois'].loc[poi, 'nombre'] if 'nombre' in data['pois'].columns else f"POI {poi}"

                coords.append((lat, lon))
                poi_names.append(name)

        # Agregar marcadores para cada POI de esta ruta
        for i, ((lat, lon), name) in enumerate(zip(coords, poi_names)):
            # Estilo especial para origen/destino
            if i == 0 or i == len(coords) - 1:
                icon = folium.Icon(color=color, icon='home')
                popup = f"<b>Origen/Destino - Ruta #{rank}:</b> {name}"
            else:
                icon = folium.Icon(color=color, icon='info-sign')
                popup = f"<b>POI #{i} - Ruta #{rank}:</b> {name}"

            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=f"R{rank}-{i}. {name}",
                icon=icon
            ).add_to(route_layer)

        # Obtener y agregar rutas realistas entre POIs
        for i in range(len(coords) - 1):
            # Obtener coordenadas de origen y destino
            origin = coords[i]
            destination = coords[i + 1]

            # Usar OSRM (OpenStreetMap Routing Machine) para obtener la ruta
            try:
                # Formato: lon,lat (OSRM usa longitud primero)
                url = f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=full&geometries=polyline"
                response = requests.get(url)

                if response.status_code == 200:
                    route_data = response.json()

                    if 'routes' in route_data and len(route_data['routes']) > 0:
                        # Decodificar la polyline
                        route_coords = polyline.decode(route_data['routes'][0]['geometry'])

                        # Agregar la ruta al mapa
                        folium.PolyLine(
                            locations=route_coords,
                            color=color,
                            weight=4,
                            opacity=0.7,
                            tooltip=f"Ruta #{rank}: {poi_names[i]} → {poi_names[i+1]}"
                        ).add_to(route_layer)
                    else:
                        print(f"    No se encontró ruta para R{rank} segmento {i+1}")
                else:
                    print(f"    Error en la API para R{rank} segmento {i+1}: {response.status_code}")

                # Esperar un poco para no sobrecargar la API
                time.sleep(0.5)

            except Exception as e:
                print(f"    Error obteniendo ruta para R{rank} segmento {i+1}: {e}")
                # Crear línea directa como fallback
                folium.PolyLine(
                    locations=[[origin[0], origin[1]], [destination[0], destination[1]]],
                    color=color,
                    weight=2,
                    opacity=0.5,
                    dash_array='5',
                    tooltip=f"Conexión directa - Ruta #{rank}: {poi_names[i]} → {poi_names[i+1]}"
                ).add_to(route_layer)

        # Añadir capa al mapa
        route_layer.add_to(m)

    # Añadir control de capas
    folium.LayerControl().add_to(m)

    # Añadir leyenda al mapa
    legend_html = '''
     <div style="position: fixed;
                 bottom: 50px; right: 50px; width: 230px;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; padding: 10px;
                 border-radius: 6px;">
     <div style="font-weight: bold; margin-bottom: 10px; text-align: center;">Mejores Rutas para {}</div>
    '''.format(grupo_id)

    for i, (color, route_data) in enumerate(zip(route_colors[:len(routes)], routes)):
        rank = route_data['rank']
        score = route_data['topsis_score']
        legend_html += '''
         <div style="margin-bottom: 5px;">
         <span style="background-color:{}; width:12px; height:12px; display:inline-block;"></span>
         <span style="margin-left:5px;">Ruta #{} (Score: {:.3f})</span>
         </div>
        '''.format(color, rank, score)

    legend_html += '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))

    # Guardar mapa
    map_file = f'{grupo_id}_top{top_n}_rutas_combinadas.html'
    m.save(map_file)
    print(f"  Mapa combinado guardado como '{map_file}'")

