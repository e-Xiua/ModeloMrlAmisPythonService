"""Implementación del worker MRL-AMIS que usa dataclasses de dominio."""

import logging
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from bucle_principal_mrl_amis.bucle_iterativo_mrl_amis import (
    calculate_average_ratio_pareto, calculate_hypervolume, 
    find_pareto_front, spacing_metric, update_population
)
from estado_y_recompensa_rl.definir_comportamiento import calculate_reward, get_state
from intelligence_boxes.definir_intelligence_boxes import (
    ib_diversity_mutation, ib_guided_perturbation, ib_inversion_mutation, 
    ib_local_search, ib_random_perturbation, ib_swap_mutation
)
from grpc_queue.data_mappers import build_domain_payload
from runmodel.models.data_generator import generate_synthetic_data
from AgenteQLearning.QLearningAgent import QLearningAgent
from generacionWorkPackages.workPackages import generar_work_packages, decodificar_wp
from generacionWorkPackages.funcionesObjetivo import evaluar_funciones_objetivo

logger = logging.getLogger(__name__)


class DomainMRLAMISWorker:
    """Worker que usa dataclasses de dominio para ejecutar MRL-AMIS."""
    
    def __init__(self):
        """Inicializar el worker de optimización de rutas"""
        logger.info("Inicializando DomainMRLAMISWorker...")
        
        # Configuración predeterminada del modelo MRL-AMIS
        self.default_config = {
            'num_work_packages': 100,
            'max_iterations': 100,
            'num_pois': 15,
            'max_pois_per_route': 100000,
            'min_pois_per_route': 2,
            'num_routes_per_wp': 3,
            'max_duration_per_route': 720,
            'maximize_objectives_list': [True, False, False, True, False],
            'ref_point_hypervolume': [0, 1000, 1000, -1000, 1000]
        }
        
        # Configuración del agente RL
        self.rl_config = {
            'state_space_size': 7,
            'action_space_size': 6,
            'learning_rate': 0.2,
            'discount_factor': 0.85,
            'epsilon_start': 1.0,
            'epsilon_decay_rate': 0.995,
            'min_epsilon': 0.15
        }
        
        # Intelligence Boxes disponibles
        self.intelligence_boxes = {
            0: ib_random_perturbation,
            1: ib_swap_mutation,
            2: ib_inversion_mutation,
            3: ib_guided_perturbation,
            4: ib_local_search,
            5: ib_diversity_mutation
        }
        
        logger.info("DomainMRLAMISWorker inicializado correctamente")

    def execute_optimization(
        self,
        job_data: Dict[str, Any], 
        request_data: Dict[str, Any], 
        job_id: str, 
        progress_callback=None
    ) -> Dict[str, Any]:
        """Ejecuta optimización usando el mapeo completo a dataclasses."""
        
        try:
            start_time = time.time()
            logger.info(f"Iniciando optimización MRL-AMIS con dataclasses para trabajo {job_id}")
            
            if progress_callback:
                progress_callback(job_id, 5.0, "Inicializando mapeo de datos...")

            # 1. Mapear request_data a dataclasses de dominio
            domain_payload = build_domain_payload(
                job_data=job_data,
                request_data=request_data,
                default_group_id=f"group_{job_id[:8]}"
            )
            
            if progress_callback:
                progress_callback(job_id, 15.0, f"Mapeados {len(domain_payload.pois)} POIs y grupo turístico")

            # 2. Generar datos sintéticos adicionales si es necesario
            synthetic_data = generate_synthetic_data(num_pois=len(domain_payload.pois), seed=42)
            
            if progress_callback:
                progress_callback(job_id, 25.0, "Generados datos sintéticos complementarios")

            # 3. Estructuras de datos para MRL-AMIS
            data = self._prepare_mrl_amis_data(domain_payload, synthetic_data)
            
            if progress_callback:
                progress_callback(job_id, 35.0, "Preparadas estructuras MRL-AMIS")

            # 4. Información del grupo adaptada
            grupo_info = self._create_grupo_info_from_domain(domain_payload)
            data['grupo_info'] = grupo_info
            
            if progress_callback:
                progress_callback(job_id, 45.0, "Configurado grupo turístico")

            # 5. Ejecutar algoritmo MRL-AMIS
            result = self._execute_mrl_amis_core(
                data, domain_payload, progress_callback, job_id
            )
            
            execution_time = time.time() - start_time
            result['execution_time_seconds'] = execution_time
            result['generated_at'] = datetime.now().isoformat()
            
            if progress_callback:
                progress_callback(job_id, 100.0, "Optimización completada exitosamente")
            
            logger.info(f"Optimización MRL-AMIS completada para trabajo {job_id} en {execution_time:.2f} segundos")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización MRL-AMIS para trabajo {job_id}: {str(e)}")
            if progress_callback:
                progress_callback(job_id, 0.0, f"Error: {str(e)}")
            raise e

    def _prepare_mrl_amis_data(
        self, 
        domain_payload, 
        synthetic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepara las estructuras de datos necesarias para MRL-AMIS."""
        
        # Usar DataFrames del domain_payload
        data = {
            'pois': domain_payload.pois_dataframe,
            'distances': domain_payload.distance_matrix,
            'travel_times': domain_payload.travel_time_matrix,
            'preferencias_grupos_turistas': domain_payload.group_preferences_matrix,
        }
        
        logger.info(
            f"DataFrame POIs: {data['pois'].shape}, "
            f"Distance Matrix: {data['distances'].shape}, "
            f"Travel Time Matrix: {data['travel_times'].shape}"
        )
        
        # Agregar matrices sintéticas adicionales del generador
        for key in ['costs', 'co2_emission_cost', 'accident_risk', 'costos_experiencia']:
            if key in synthetic_data:
                data[key] = synthetic_data[key]
                
            # COSTOS DE EXPERIENCIA: Crear tabla con estructura correcta
        if 'costos_experiencia' in synthetic_data:
            costos_exp = synthetic_data['costos_experiencia']
            
            # Si es DataFrame, verificar que tenga los POI IDs como índice
            if isinstance(costos_exp, pd.DataFrame):
                # Reindexar si es necesario
                poi_ids = [poi.id for poi in domain_payload.pois]
                if list(costos_exp.index) != poi_ids:
                    logger.info(f"Reindexando costos_experiencia para coincidir con POI IDs")
                    # Crear nuevo DataFrame con índices correctos
                    costos_exp_reindexed = pd.DataFrame(
                        costos_exp.values[:len(poi_ids)],
                        index=poi_ids,
                        columns=costos_exp.columns
                    )
                    data['costos_experiencia'] = costos_exp_reindexed
                else:
                    data['costos_experiencia'] = costos_exp
                
                logger.info(f"Costos de experiencia agregados:")
                logger.info(f"  -> Shape: {data['costos_experiencia'].shape}")
                logger.info(f"  -> Índices (POI IDs): {data['costos_experiencia'].index.tolist()}")
                logger.info(f"  -> Columnas: {data['costos_experiencia'].columns.tolist()}")
                
        params = synthetic_data.get('parametros_generales')
        if params is not None and isinstance(params, dict):
            data['parametros_generales'] = pd.DataFrame.from_dict(params, orient='index', columns=['Valor'])
        else:
            logger.warning("⚠️  PARÁMETROS GENERALES NO DISPONIBLES")
        
        # Parámetros generales
        data['parametros_generales'] = synthetic_data.get('parametros_generales', {})
        logger.info(f"Parámetros generales agregados: {len(data['parametros_generales'])} parámetros")
        logger.info(f"Parámetros generales: {data['parametros_generales']}")
        
        return data

    def _create_grupo_info_from_domain(self, domain_payload) -> Dict[str, Any]:
        """Crea información del grupo basada en el TouristGroup."""
        
        group = domain_payload.tourist_group

        return {
            'tiempo_disponible': group.tiempo_disponible_min,
            'presupuesto': group.presupuesto_max_usd,
            'tipo_turista': group.tipo_turista,
            'sostenibilidad_min': group.sostenibilidad_min,
            'nivel_aventura': group.nivel_aventura,
            'conciencia_ambiental': group.conciencia_ambiental,
            'tolerancia_riesgo': group.tolerancia_riesgo,
            'origen': group.origen,
            'destino': group.destino,
            'min_pois_per_route': 1,
            'max_pois_per_route': min(10, len(domain_payload.pois)),
            'preferencias_tipos': group.preferencias_tipos
        }

    def _log_input_data(
        self, 
        domain_payload, 
        data: Dict[str, Any], 
        job_id: str
    ):
        """Registra detalladamente los datos de entrada."""
        
        logger.info("=" * 80)
        logger.info(f"INICIANDO MRL-AMIS CORE - JOB ID: {job_id}")
        logger.info("=" * 80)
        
        # Log POIs
        logger.info(f"\nPOIs RECIBIDOS ({len(domain_payload.pois)} total)")
        for i, poi in enumerate(domain_payload.pois, 1):
            logger.info(
                f"POI {i}: ID={poi.id}, Nombre={poi.name}, "
                f"Pos=({poi.position[0]:.4f}, {poi.position[1]:.4f})"
            )


        # Log grupo turístico
        group = domain_payload.tourist_group
        logger.info(f"\nGRUPO TURÍSTICO:")
        logger.info(f"  ID: {group.grupo_id}")
        logger.info(f"  Tipo: {group.tipo_turista}")
        logger.info(f"  Presupuesto: ${group.presupuesto_max_usd}")
        logger.info(f"  Tiempo: {group.tiempo_disponible_min} min")
        logger.info(f"  Origen: {group.origen} -> Destino: {group.destino}")
        
        # Log matrices
        self._log_matrix(data.get('distances'), "DISTANCIAS (km)")
        self._log_matrix(data.get('travel_times'), "TIEMPOS DE VIAJE (min)")
        self._log_matrix(data.get('costs'), "COSTOS (USD)")
        self._log_matrix(data.get('co2_emission_cost'), "EMISIONES CO2 (kg)")
        self._log_matrix(data.get('accident_risk'), "RIESGO DE ACCIDENTES")
        self._log_matrix(data.get('preferencias_grupos_turistas'), "Preferencias Grupo-POI")
        self._log_matrix(data.get('costos_experiencia'), "COSTOS DE EXPERIENCIA (USD)")

    def _log_matrix(self, matrix, title: str):
        """Registra una matriz con estadísticas."""
        
        logger.info(f"\n{'='*80}")
        logger.info(title)
        logger.info(f"{'='*80}")
        
        if matrix is not None:
            logger.info(f"Dimensiones: {matrix.shape}")
            if hasattr(matrix, 'to_string'):
                logger.info(f"\n{matrix.to_string()}")
                
                # Estadísticas (excluyendo diagonal)
                non_diag = matrix.where(matrix > 0)
                logger.info(f"\nEstadísticas:")
                logger.info(f"  Mínimo: {non_diag.min().min():.2f}")
                logger.info(f"  Máximo: {matrix.max().max():.2f}")
                logger.info(f"  Promedio: {matrix.mean().mean():.2f}")
        else:
            logger.warning(f"⚠️  {title} NO DISPONIBLE")

    def _execute_mrl_amis_core(
        self,
        data: Dict[str, Any], 
        domain_payload, 
        progress_callback, 
        job_id: str
    ) -> Dict[str, Any]:
        """Ejecuta la lógica central del modelo MRL-AMIS."""
        
        # Log detallado de entrada
        self._log_input_data(domain_payload, data, job_id)
        
        # Configuración del modelo
        num_work_packages = self.default_config['num_work_packages']
        max_iterations = self.default_config['max_iterations']
        maximize_objectives_list = self.default_config['maximize_objectives_list']
        ref_point_hypervolume = self.default_config['ref_point_hypervolume']
        
        # Información del grupo
        grupo_info = data['grupo_info']
        tipo_turista = grupo_info.get('tipo_turista', 'nacional')
        origen_poi = grupo_info.get('origen', '1')
        
        # Generar Work Packages
        num_pois = len(domain_payload.pois)
        wps_df = generar_work_packages(q=num_work_packages, d=num_pois)
        
        if progress_callback:
            progress_callback(job_id, 50.0, f"Generados {len(wps_df)} Work Packages")
        
        logger.info(f"Generados {len(wps_df)} Work Packages de dimensión {num_pois}")
        
        # Evaluar población inicial
        resultados_iniciales = self._evaluate_initial_population(
            wps_df, data, grupo_info, tipo_turista, origen_poi, 
            job_id, progress_callback
        )
        
        num_factibles = sum(1 for sol in resultados_iniciales if sol.get('is_feasible', False))
        logger.info(f"Población inicial: {num_factibles}/{len(resultados_iniciales)} factibles")
        
        # Inicializar agente RL
        rl_agent = self._initialize_rl_agent()
        
        # Ejecutar bucle principal
        final_results = self._run_mrl_amis_loop(
            resultados_iniciales,
            rl_agent,
            data,
            grupo_info,
            tipo_turista,
            origen_poi,
            max_iterations,
            maximize_objectives_list,
            ref_point_hypervolume,
            num_work_packages,
            job_id,
            progress_callback
        )
        
        # Formatear y retornar resultados
        return self._format_final_results(
            final_results,
            domain_payload,
            max_iterations,
            job_id,
            progress_callback
        )

    def _evaluate_initial_population(
        self,
        wps_df,
        data: Dict[str, Any],
        grupo_info: Dict[str, Any],
        tipo_turista: str,
        origen_poi: str,
        job_id: str,
        progress_callback
    ) -> list:
        """Evalúa la población inicial de Work Packages."""
        
        resultados = []
        total_wps = len(wps_df)
        
        logger.info(f"Evaluando {total_wps} Work Packages iniciales...")
        
        for i, (wp_name, wp_series) in enumerate(wps_df.iterrows()):
            wp_vector = wp_series.values
            
            # Decodificar WP
            ruta_decodificada, is_feasible = decodificar_wp(
                wp_vector=wp_vector,
                pois_df=data["pois"],
                travel_times_df=data["travel_times"],
                grupo_info=grupo_info,
                origen=origen_poi
            )
            
            # Evaluar objetivos
            objetivos = evaluar_funciones_objetivo(
                ruta_decodificada, 
                data, 
                tipo_turista=tipo_turista
            )
            
            # VALIDACIÓN ROBUSTA
            is_truly_feasible = is_feasible and objetivos is not None and all(
                key in objetivos for key in [
                    'preferencia_total', 'costo_total', 'co2_total',
                    'sust_ambiental', 'sust_economica', 'sust_social', 'riesgo_total'
                ]
            )
            
            resultados.append({
                'wp_name': wp_name,
                'wp_original': wp_vector,
                'ruta_decodificada': ruta_decodificada,
                'objetivos': objetivos,
                'is_feasible': is_truly_feasible
            })
            
            # Progreso
            if (i + 1) % 20 == 0 and progress_callback:
                progress = 50.0 + ((i + 1) / total_wps) * 10.0
                progress_callback(job_id, progress, f"Evaluados {i+1}/{total_wps} WPs")
        
        return resultados

    def _initialize_rl_agent(self) -> QLearningAgent:
        """Inicializa el agente de Reinforcement Learning."""
        
        return QLearningAgent(
            state_space_size=self.rl_config['state_space_size'],
            action_space_size=self.rl_config['action_space_size'],
            learning_rate=self.rl_config['learning_rate'],
            discount_factor=self.rl_config['discount_factor'],
            epsilon=self.rl_config['epsilon_start'],
            epsilon_decay_rate=self.rl_config['epsilon_decay_rate'],
            min_epsilon=self.rl_config['min_epsilon']
        )

    def _run_mrl_amis_loop(
        self,
        current_population_results: list,
        rl_agent: QLearningAgent,
        data: Dict[str, Any],
        grupo_info: Dict[str, Any],
        tipo_turista: str,
        origen_poi: str,
        max_iterations: int,
        maximize_objectives_list: list,
        ref_point_hypervolume: list,
        num_work_packages: int,
        job_id: str,
        progress_callback
    ) -> Dict[str, Any]:
        """Ejecuta el bucle principal de MRL-AMIS."""
        
        hypervolume_history = []
        iteration_metrics = {
            'iteration': [],
            'pareto_front_size': [],
            'hypervolume': [],
            'feasible_count': []
        }
        
        logger.info(f"Iniciando bucle MRL-AMIS con {max_iterations} iteraciones...")
        
        for iteration in range(max_iterations):
            # Progreso
            progress = 60.0 + ((iteration + 1) / max_iterations) * 30.0
            if progress_callback and (iteration % 5 == 0):
                progress_callback(job_id, progress, f"Iteración {iteration+1}/{max_iterations}")
            
            # Extraer objetivos actuales
            current_objectives = self._extract_objectives(
                current_population_results, 
                only_feasible=True
            )
            
            # Encontrar frente de Pareto
            pareto_front = []
            if len(current_objectives) > 0:
                pareto_front, _ = find_pareto_front(current_objectives, maximize_objectives_list)
            
            # Generar nuevas soluciones
            new_solutions = self._generate_new_solutions(
                current_population_results,
                rl_agent,
                data,
                grupo_info,
                tipo_turista,
                origen_poi,
                pareto_front,
                maximize_objectives_list,
                hypervolume_history
            )
            
            # Actualizar población
            current_population_results = update_population(
                current_population_results,
                new_solutions,
                num_work_packages,
                maximize_objectives_list
            )
            
            # Calcular métricas
            metrics = self._calculate_iteration_metrics(
                current_population_results,
                maximize_objectives_list,
                ref_point_hypervolume,
                iteration + 1
            )
            
            # Almacenar métricas
            for key, value in metrics.items():
                iteration_metrics[key].append(value)
            
            if metrics['hypervolume'] > 0:
                hypervolume_history.append(metrics['hypervolume'])
            
            # Log cada 10 iteraciones
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iter {iteration+1}: Factibles={metrics['feasible_count']}, "
                    f"Pareto={metrics['pareto_front_size']}, HV={metrics['hypervolume']:.4f}"
                )
            
            # Decaimiento epsilon
            rl_agent.decay_epsilon_with_restart()
        
        return {
            'population': current_population_results,
            'hypervolume_history': hypervolume_history,
            'iteration_metrics': iteration_metrics
        }

    def _extract_objectives(
        self, 
        solutions: list, 
        only_feasible: bool = False
    ) -> np.ndarray:
        """Extrae objetivos de las soluciones."""
        
        filtered = solutions
        if only_feasible:
            filtered = [s for s in solutions if s.get('is_feasible', False)]
        
        objectives = np.array([
            [
                obj.get('preferencia_total', 0),
                obj.get('costo_total', 0),
                obj.get('co2_total', 0),
                obj.get('sust_ambiental', 0) + obj.get('sust_economica', 0) + obj.get('sust_social', 0),
                obj.get('riesgo_total', 0)
            ]
            for sol in filtered
            for obj in [sol.get('objetivos', {})]
            if obj is not None
        ])
        
        return objectives

    def _generate_new_solutions(
        self,
        population: list,
        rl_agent: QLearningAgent,
        data: Dict[str, Any],
        grupo_info: Dict[str, Any],
        tipo_turista: str,
        origen_poi: str,
        pareto_front: list,
        maximize_objectives_list: list,
        hypervolume_history: list
    ) -> list:
        """Genera nuevas soluciones usando Intelligence Boxes."""
        
        new_solutions = []
        
        for original_solution in population:
            # Estado actual
            current_state = get_state(
                original_solution,
                pareto_front,
                maximize_objectives_list,
                hypervolume_history
            )
            
            # Elegir Intelligence Box
            action_idx = rl_agent.choose_action(current_state)
            ib_func = self.intelligence_boxes.get(action_idx, ib_random_perturbation)
            
            # Aplicar IB
            original_wp = original_solution.get('wp_original', np.array([]))
            
            kwargs = {
                'grupo_info': grupo_info,
                'pois_df': data["pois"],
                'travel_times_df': data["travel_times"],
                'data': data,
                'tipo_turista': tipo_turista,
                'maximize_objectives_list': maximize_objectives_list,
                'current_pareto_front': pareto_front,
                'current_population_results': population,
                'origen': origen_poi
            }
            
            try:
                modified_wp = ib_func(original_wp, **kwargs)
            except Exception as e:
                logger.warning(f"Error en IB {action_idx}: {e}")
                modified_wp = original_wp.copy()
            
            # Evaluar nueva solución
            decoded_route, is_feasible = decodificar_wp(
                wp_vector=modified_wp,
                pois_df=data["pois"],
                travel_times_df=data["travel_times"],
                grupo_info=grupo_info,
                origen=origen_poi
            )
            
            metrics = evaluar_funciones_objetivo(decoded_route, data, tipo_turista)
            
            # Validar factibilidad
            is_truly_feasible = is_feasible and metrics is not None and all(
                key in metrics for key in [
                    'preferencia_total', 'costo_total', 'co2_total',
                    'sust_ambiental', 'sust_economica', 'sust_social', 'riesgo_total'
                ]
            )
            
            new_solution = {
                'wp_original': modified_wp,
                'ruta_decodificada': decoded_route,
                'objetivos': metrics,
                'is_feasible': is_truly_feasible
            }
            
            # Aprender
            reward = calculate_reward(
                original_solution,
                new_solution,
                pareto_front,
                maximize_objectives_list
            )
            
            next_state = get_state(
                new_solution,
                pareto_front,
                maximize_objectives_list,
                hypervolume_history
            )
            
            rl_agent.learn(current_state, action_idx, reward, next_state)
            
            new_solutions.append(new_solution)
        
        return new_solutions

    def _calculate_iteration_metrics(
        self,
        population: list,
        maximize_objectives_list: list,
        ref_point: list,
        iteration_num: int
    ) -> Dict[str, Any]:
        """Calcula métricas de la iteración."""
        
        feasible_count = sum(1 for s in population if s.get('is_feasible', False))
        
        objectives = self._extract_objectives(population, only_feasible=True)
        
        metrics = {
            'iteration': iteration_num,
            'feasible_count': feasible_count,
            'pareto_front_size': 0,
            'hypervolume': 0.0
        }
        
        if len(objectives) > 0:
            pareto_front, _ = find_pareto_front(objectives, maximize_objectives_list)
            metrics['pareto_front_size'] = len(pareto_front)
            
            if len(pareto_front) > 0:
                try:
                    hv = calculate_hypervolume(pareto_front, ref_point, maximize_objectives_list)
                    metrics['hypervolume'] = hv
                except Exception as e:
                    logger.warning(f"Error calculando HV: {e}")
        
        return metrics

    def _format_final_results(
        self,
        mrl_amis_results: Dict[str, Any],
        domain_payload,
        max_iterations: int,
        job_id: str,
        progress_callback
    ) -> Dict[str, Any]:
        """Formatea los resultados finales."""
        
        if progress_callback:
            progress_callback(job_id, 95.0, "Formateando resultados finales...")
        
        population = mrl_amis_results['population']
        
        # Obtener mejor solución
        feasible = [s for s in population if s.get('is_feasible', False) and s.get('objetivos')]
        
        best_solution = None
        if feasible:
            # Ordenar por preferencia total
            feasible.sort(
                key=lambda x: x['objetivos'].get('preferencia_total', 0),
                reverse=True
            )
            best_solution = feasible[0]
        
        # Crear solución por defecto si no hay factibles
        if not best_solution:
            logger.warning("No hay soluciones factibles. Creando ruta básica...")
            best_solution = self._create_fallback_solution(domain_payload)
        
        # Formatear secuencia
        optimized_sequence = self._format_optimized_sequence(
            best_solution,
            domain_payload
        )
        
        # Construir respuesta
        objectives = best_solution['objetivos']
        
        result = {
            'optimized_sequence': optimized_sequence,
            'route_description': f'Ruta optimizada con {len(optimized_sequence)} POIs usando MRL-AMIS',
            'total_distance_km': objectives.get('distancia_total', 0.0),
            'total_time_minutes': int(objectives.get('tiempo_total', 0)),
            'total_cost': objectives.get('costo_total', 0.0),
            'optimization_algorithm': 'MRL-AMIS-Real',
            'optimization_score': objectives.get('preferencia_total', 0) / 100.0,
            'iterations_completed': max_iterations,
            'feasible_solutions_found': len(feasible),
            'pareto_front_size': mrl_amis_results['iteration_metrics']['pareto_front_size'][-1] if mrl_amis_results['iteration_metrics']['pareto_front_size'] else 0,
            'final_hypervolume': mrl_amis_results['hypervolume_history'][-1] if mrl_amis_results['hypervolume_history'] else 0.0,
            'used_dataclasses': True,
            'tourist_group_id': domain_payload.tourist_group.grupo_id,
            'tourist_group_type': domain_payload.tourist_group.tipo_turista,
            'iteration_metrics': mrl_amis_results['iteration_metrics']
        }
        
        logger.info(
            f"Resultado final - Score: {result['optimization_score']:.2f}, "
            f"Distancia: {result['total_distance_km']:.1f}km, "
            f"Tiempo: {result['total_time_minutes']}min"
        )
        
        return result

    def _create_fallback_solution(self, domain_payload) -> Dict[str, Any]:
        """Crea una solución de respaldo básica."""
        
        first_poi = domain_payload.pois[0]
        last_poi = domain_payload.pois[-1] if len(domain_payload.pois) > 1 else first_poi
        
        return {
            'ruta_decodificada': [first_poi.id, last_poi.id],
            'objetivos': {
                'preferencia_total': 75.0,
                'costo_total': sum(p.entry_cost + p.stay_cost_per_hour for p in domain_payload.pois[:2]),
                'tiempo_total': sum(p.duracion_visita_min for p in domain_payload.pois[:2]),
                'distancia_total': 25.0,
                'co2_total': 5.0,
                'sust_ambiental': 50.0,
                'sust_economica': 50.0,
                'sust_social': 50.0,
                'riesgo_total': 0.1
            },
            'is_feasible': True
        }

    def _format_optimized_sequence(
        self, 
        solution: Dict[str, Any],
        domain_payload
    ) -> list:
        """Formatea la secuencia optimizada para la respuesta."""
        
        sequence = []
        route = solution['ruta_decodificada']
        
        for poi_id in route:
            # Buscar POI
            matching_poi = next((p for p in domain_payload.pois if p.id == poi_id), None)
            
            if matching_poi:
                external_id = domain_payload.internal_to_external_ids.get(poi_id, poi_id)
                
                sequence.append({
                    'poi_id': int(external_id) if external_id.isdigit() else hash(external_id) % 10000,
                    'name': matching_poi.name,
                    'latitude': matching_poi.position[0],
                    'longitude': matching_poi.position[1],
                    'visit_order': len(sequence) + 1,
                    'estimated_visit_time': matching_poi.duracion_visita_min,
                    'arrival_time': f"{8 + len(sequence)}:00",
                    'departure_time': f"{8 + len(sequence) + 1}:30"
                })
        
        return sequence
