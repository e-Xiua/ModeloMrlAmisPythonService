import numpy as np

class QLearningAgent:
    """
    Agente de Q-learning mejorado para seleccionar los mejores Intelligence Boxes
    con soporte para diversidad y exploración sostenida.
    """
    def __init__(self, state_space_size, action_space_size, learning_rate=0.2,
                 discount_factor=0.85, epsilon=1.0, epsilon_decay_rate=0.995, min_epsilon=0.15):
        """
        Inicializa el agente de Q-learning con parámetros optimizados.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate  # Alpha - tasa de aprendizaje
        self.discount_factor = discount_factor  # Gamma - factor de descuento
        self.epsilon = epsilon  # Epsilon para política epsilon-greedy
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Inicializar tabla Q con valores pequeños aleatorios en lugar de ceros
        # Esto ayuda a romper empates y fomenta la exploración inicial
        self.q_table = np.random.uniform(low=0.0, high=0.01,
                                         size=(state_space_size, action_space_size))

        # Contador de uso de cada acción para cada estado
        self.action_count = np.zeros((state_space_size, action_space_size))

        # Variables para seguimiento
        self.cumulative_reward = 0
        self.last_rewards = []  # Últimas n recompensas

        # Contador para reinicios periódicos de epsilon
        self.iteration_count = 0

        # Para seguimiento de diversidad
        self.diversity_boosts = 0

        print(f"Inicializado agente QLearning con tabla Q de forma: {self.q_table.shape}")
        print(f"Parámetros: α={learning_rate}, γ={discount_factor}, ε={epsilon}, ε_decay={epsilon_decay_rate}, ε_min={min_epsilon}")

    def choose_action(self, state):
        """
        Elige una acción usando política epsilon-greedy con un componente de exploración adicional.
        """
        # Verificar estado válido
        if state < 0 or state >= self.state_space_size:
            print(f"Advertencia: Estado {state} inválido. Usando estado 0.")
            state = 0

        # Estrategia epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Exploración: elegir acción aleatoria
            # Si estamos en estado de estancamiento (estado 6), favorecer operadores de diversidad
            if state == 6 and np.random.rand() < 0.7:  # 70% de probabilidad en estado de estancamiento
                # Operadores que favorecen diversidad
                diversity_operators = [2, 3, 5]  # Índices de inversion, guided, diversity
                # Filtrar operadores válidos
                valid_operators = [op for op in diversity_operators if op < self.action_space_size]
                if valid_operators:
                    action = np.random.choice(valid_operators)
                    self.diversity_boosts += 1
                else:
                    action = np.random.randint(self.action_space_size)
            else:
                action = np.random.randint(self.action_space_size)
        else:
            # Explotación: elegir acción con mayor valor Q
            # En caso de empate, elegir la menos utilizada
            q_values = self.q_table[state, :]
            max_q = np.max(q_values)

            # Identificar acciones con valores Q máximos
            max_actions = np.where(q_values == max_q)[0]

            if len(max_actions) > 1:
                # Si hay empate, elegir la acción menos utilizada
                action_counts = self.action_count[state, max_actions]
                action = max_actions[np.argmin(action_counts)]
            else:
                action = max_actions[0]

        # Incrementar contador de uso
        self.action_count[state, action] += 1

        return action

    def learn(self, state, action, reward, next_state):
      """
      Actualiza la tabla Q basada en la experiencia observada con manejo de
      errores mejorado y soporte para diversidad en situaciones de estancamiento.
      """
    # Verificar parámetros válidos
      if not (0 <= state < self.state_space_size and
              0 <= next_state < self.state_space_size and
              0 <= action < self.action_space_size):
          print(f"Error en parámetros de learn: estado={state}, acción={action}, siguiente_estado={next_state}")
          return

    # Registrar recompensa para seguimiento
      self.cumulative_reward += reward
      self.last_rewards.append(reward)
      if len(self.last_rewards) > 20:
          self.last_rewards.pop(0)

    # Ecuación de actualización de Q-learning
      current_q = self.q_table[state, action]
      max_next_q = np.max(self.q_table[next_state, :])

    # Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
      new_q = current_q + self.learning_rate * (
          reward + self.discount_factor * max_next_q - current_q
      )

      self.q_table[state, action] = new_q

    # Incrementar contador de uso
      self.action_count[state, action] += 1

    # Incentivo adicional para diversidad en caso de estancamiento (estado 6)
      if state == 6:  # Estado de estancamiento
          # Incrementar valor Q para operadores de diversidad
          diversity_operators = [2, 3, 5]  # Índices de inversion, guided, diversity
          for op in diversity_operators:
              if op < self.action_space_size:
                  # Pequeño incremento para estos operadores en caso de estancamiento
                  self.q_table[state, op] += 0.05

        # Pequeña penalización para operadores de explotación local
          exploitation_operators = [0, 4]  # Índices de operadores de explotación local
          for op in exploitation_operators:
              if op < self.action_space_size:
                  # Pequeña penalización para evitar estancamiento
                  self.q_table[state, op] = max(0.0, self.q_table[state, op] - 0.02)

    def decay_epsilon(self):
        """
        Reduce epsilon para disminuir exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)
        return self.epsilon

    def decay_epsilon_with_restart(self):
        """
        Decae epsilon con reinicios periódicos para fomentar exploración.
        """
        # Guardar valor anterior para reporte
        old_epsilon = self.epsilon

        # Decaimiento normal
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

        # Reiniciar epsilon cada 20 iteraciones para fomentar exploración
        if self.iteration_count % 20 == 0 and self.iteration_count > 0:
            self.epsilon = min(0.5, self.epsilon * 2)  # Reiniciar a 0.5 o duplicar el valor actual
            print(f"Reinicio periódico de epsilon: {old_epsilon:.4f} → {self.epsilon:.4f}")

        self.iteration_count += 1

        return self.epsilon

    def reset_exploration_if_needed(self, iteration, threshold=20):
        """
        Reinicia parcialmente la exploración si la recompensa promedio reciente es baja.
        """
        if len(self.last_rewards) >= threshold:
            avg_reward = sum(self.last_rewards) / len(self.last_rewards)

            # Si la recompensa promedio es baja, aumentar exploración
            if avg_reward < 1.0:
                old_epsilon = self.epsilon
                self.epsilon = min(0.5, self.epsilon * 1.5)

                if self.epsilon > old_epsilon:
                    print(f"Reiniciando exploración en iteración {iteration}. "
                          f"Epsilon: {old_epsilon:.3f} → {self.epsilon:.3f}")

    def get_best_action(self, state):
        """
        Devuelve la mejor acción para un estado (sin exploración).
        """
        if state < 0 or state >= self.state_space_size:
            state = 0
        return np.argmax(self.q_table[state, :])

    def get_q_table(self):
        """
        Devuelve la tabla Q actual.
        """
        return self.q_table

    def get_policy(self):
        """
        Devuelve la política actual como un mapeo de estado a mejor acción.
        """
        return {state: np.argmax(self.q_table[state, :])
                for state in range(self.state_space_size)}

    def get_performance_metrics(self):
        """
        Devuelve métricas de rendimiento del agente.
        """
        return {
            "epsilon": self.epsilon,
            "cumulative_reward": self.cumulative_reward,
            "avg_recent_reward": sum(self.last_rewards) / max(1, len(self.last_rewards)),
            "action_distribution": self.action_count.sum(axis=0) / max(1, self.action_count.sum()),
            "diversity_boosts": self.diversity_boosts
        }

    def analyze_state_action_distribution(self):
        """
        Analiza la distribución de acciones por estado para entender el comportamiento del agente.
        """
        print("\nDistribución de acciones por estado:")
        for state in range(self.state_space_size):
            total_visits = self.action_count[state].sum()
            if total_visits > 0:
                print(f"Estado {state}: {total_visits} visitas")
                for action in range(self.action_space_size):
                    count = self.action_count[state, action]
                    percentage = (count / total_visits) * 100 if total_visits > 0 else 0
                    q_value = self.q_table[state, action]
                    print(f"  Acción {action}: {count} usos ({percentage:.1f}%), Q-valor = {q_value:.3f}")
            else:
                print(f"Estado {state}: No visitado")

        # Mostrar comportamiento en estado de estancamiento (si existe)
        if self.state_space_size > 6:
            print("\nComportamiento en estado de estancamiento (estado 6):")
            stagnation_q = self.q_table[6]
            stagnation_actions = np.argsort(stagnation_q)[::-1]  # Ordenar por valor Q descendente
            for action in stagnation_actions:
                print(f"  Acción {action}: Q-valor = {stagnation_q[action]:.3f}")