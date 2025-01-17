import numpy as np
from pcgym import make_env
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Container for simulation results."""
    observed_states: Dict[str, List[float]]
    disturbances: Dict[str, List[float]]
    actions: Dict[str, List[float]]
    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        """Makes SimulationResult iterable, yielding (observed_states, disturbances, actions)"""
        return iter((self.observed_states, self.disturbances, self.actions))

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int
    tsim: int
    noise_percentage: float = 0.01

    
class CSTRSimulator():
    def __init__(
        self,
        config
    ):
        """
        Initialize the CSTR simulator.
        
        Args:
            T (int): Number of time steps
            tsim (int): Simulation time period
            noise_percentage (float): Noise level for simulation
        """
        self.config = config
        
        # Define spaces
        self.action_space = {
            'low': np.array([295]),
            'high': np.array([302])
        }
        
        self.observation_space = {
            'low': np.array([0, 300, 0.8]),
            'high': np.array([1, 350, 0.9])
        }
        
        self.uncertainty_percentages = {
            'q': 0.1, 'V': 0.1, 
            'rho': 0.1, 'C': 0.1, 
            'EA_over_R': 0.1, 'k0': 0.1, 
            'UA': 0.1
        }
        
        self.uncertainty_space = {
            'low': np.array([0]*7),
            'high': np.array([1]*7)
        }

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for the simulation."""
        num_changes = np.random.randint(0, 5)
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
        setpoints = []
        current_setpoint = np.random.rand()
        
        for t in range(self.config.tsim):
            if len(change_points) > 0 and t == change_points[0]:
                current_setpoint = np.random.rand()
                change_points = change_points[1:]
            setpoints.append(current_setpoint)
        
        return {'Ca': setpoints}

    def generate_disturbances(self) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
        """Generate random disturbances for the simulation."""
        nominal_temp = 335
        nominal_conc = 0.75
        
        # Temperature disturbances
        num_changes_temp = np.random.randint(0, 4)
        change_points_temp = np.sort(np.random.choice(range(1, self.config.tsim), num_changes_temp, replace=False))
        disturbances_temp = []
        current_disturbance_temp = nominal_temp + (np.random.rand() - 0.5) * 10
        
        for t in range(self.config.tsim):
            if len(change_points_temp) > 0 and t == change_points_temp[0]:
                current_disturbance_temp = nominal_temp + (np.random.rand() - 0.5) * 10
                change_points_temp = change_points_temp[1:]
            disturbances_temp.append(current_disturbance_temp)
        
        # Concentration disturbances
        num_changes_conc = np.random.randint(0, 4)
        change_points_conc = np.sort(np.random.choice(range(1, self.config.tsim), num_changes_conc, replace=False))
        disturbances_conc = []
        current_disturbance_conc = nominal_conc + (np.random.rand() - 0.5) * 0.1
        
        for t in range(self.config.tsim):
            if len(change_points_conc) > 0 and t == change_points_conc[0]:
                current_disturbance_conc = nominal_conc + (np.random.rand() - 0.5) * 0.1
                change_points_conc = change_points_conc[1:]
            disturbances_conc.append(current_disturbance_conc)
        
        disturbances = {'Ti': disturbances_temp, 'Caf': disturbances_conc}
        disturbance_space = {'low': np.array([320, 0.7]), 'high': np.array([350, 0.8])}
        
        return disturbances, disturbance_space

    def simulate(self) -> SimulationResult:
        """
        Run a single simulation.
        
        Args:
            time_steps (int): Number of time steps for simulation
            
        Returns:
            SimulationResult: Container with simulation results
        """
        setpoints = self.generate_setpoints()
        disturbances, disturbance_space = self.generate_disturbances()
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'SP': setpoints,
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': np.array([0.8, 330, 0.8]),
            'model': 'cstr',
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'disturbance_bounds': disturbance_space,
            'disturbances': disturbances,
            'normal_uncertainty': self.uncertainty_percentages,
            'uncertainty_bounds': self.uncertainty_space
        }
        
        env = make_env(env_params)
        
        observed_states = []
        disturbance_values = []
        obs, _ = env.reset()
        done = False
        actions = []
        
        num_actions = np.random.randint(0, 5)
        initial_action = np.random.uniform(self.action_space['low'], self.action_space['high'])
        initial_action = self._normalize_action(initial_action)
        
        while not done:
            if not actions:
                action = initial_action
            elif num_actions > 0:
                if np.random.rand() < 0.5:
                    action = np.random.uniform(self.action_space['low'], self.action_space['high'])
                    action = self._normalize_action(action)
                    num_actions -= 1
                else:
                    action = self._normalize_action(actions[-1])
            
            obs, _, done, _, info = env.step(action)
            
            disturbance = obs[3:5]
            uncertain_params = obs[5:]
            obs = obs[:3]
            
            obs_unnorm = self._unnormalize_observation(obs)
            disturbance_unnorm = self._unnormalize_disturbance(disturbance, disturbance_space)
            actions_unnorm = self._unnormalize_action(action)
            
            observed_states.append(obs_unnorm)
            disturbance_values.append(disturbance_unnorm)
            actions.append(actions_unnorm)
        
        return self._format_results(observed_states, disturbance_values, actions)
    
    def run_multiple_simulations(self) -> List[SimulationResult]:
        """
        Run multiple simulations and return results.
        
        Args:
            num_simulations (int): Number of simulations to run
            time_steps (int): Number of time steps per simulation
            
        Returns:
            List[SimulationResult]: List of simulation results
        """
        return [self.simulate() for _ in range(self.config.n_simulations)]

    def plot_results(self, results: List[SimulationResult]) -> None:
        """
        Plot results from multiple simulations.
        
        Args:
            results (List[SimulationResult]): List of simulation results
        """
        # Create subplots for observed states
        fig_obs, axs_obs = plt.subplots(3, 1, figsize=(10, 8))
        fig_dist, axs_dist = plt.subplots(2, 1, figsize=(10, 8))
        fig_act, axs_act = plt.subplots(1, 1, figsize=(10, 4))
        
        # Plot observed states
        for i, result in enumerate(results):
            axs_obs[0].plot(result.observed_states['Ca'], label=f'Simulation {i+1}')
            axs_obs[0].set_title('Concentration of A (Ca)')
            axs_obs[0].set_xlabel('Time')
            axs_obs[0].set_ylabel('Ca')
            
            axs_obs[1].plot(result.observed_states['T'], label=f'Simulation {i+1}')
            axs_obs[1].set_title('Temperature (T)')
            axs_obs[1].set_xlabel('Time')
            axs_obs[1].set_ylabel('T')
            
            axs_obs[2].plot(result.observed_states['Ca_s'], label=f'Simulation {i+1}')
            axs_obs[2].set_title('Setpoint - (Ca_s)')
            axs_obs[2].set_xlabel('Time')
            axs_obs[2].set_ylabel('Ca_s')
        
        # Plot disturbances
        for i, result in enumerate(results):
            axs_dist[0].plot(result.disturbances['Ti'], label=f'Simulation {i+1}')
            axs_dist[0].set_title('Inlet Temperature (Ti)')
            axs_dist[0].set_xlabel('Time')
            axs_dist[0].set_ylabel('Ti')
            
            axs_dist[1].plot(result.disturbances['Caf'], label=f'Simulation {i+1}')
            axs_dist[1].set_title('Feed Concentration (Caf)')
            axs_dist[1].set_xlabel('Time')
            axs_dist[1].set_ylabel('Caf')
        
        # Plot actions
        for i, result in enumerate(results):
            axs_act.plot(result.actions['Tc'], label=f'Simulation {i+1}')
            axs_act.set_title('Coolant Temperature (Tc)')
            axs_act.set_xlabel('Time')
            axs_act.set_ylabel('Tc')
        
        # Add legends
        for ax in axs_obs:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        for ax in axs_dist:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        axs_act.legend(loc='upper right', fontsize='small', ncol=2)
        
        plt.tight_layout()
        plt.show()

    def _normalize_action(self, action: Union[float, np.ndarray]) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        return 2 * (action - self.action_space['low']) / (
            self.action_space['high'] - self.action_space['low']
        ) - 1

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action back to original range."""
        return (action + 1) * (
            self.action_space['high'] - self.action_space['low']
        ) / 2 + self.action_space['low']

    def _unnormalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert normalized observation back to original range."""
        return (obs + 1) * (
            self.observation_space['high'] - self.observation_space['low']
        ) / 2 + self.observation_space['low']

    def _unnormalize_disturbance(self, disturbance: np.ndarray, space: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert normalized disturbance back to original range."""
        return (disturbance + 1) * (space['high'] - space['low']) / 2 + space['low']

    def _format_results(
        self,
        observed_states: List[np.ndarray],
        disturbances: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> SimulationResult:
        """Format the simulation results into a structured container."""
        obs_states = {
            'Ca': [state[0] for state in observed_states],
            'T': [state[1] for state in observed_states],
            'Ca_s': [state[2] for state in observed_states],
        }
        
        dist_states = {
            'Ti': [state[0] for state in disturbances],
            'Caf': [state[1] for state in disturbances],
        }
        
        action_states = {
            'Tc': [state[0] for state in actions]
        }
        
        return SimulationResult(obs_states, dist_states, action_states)

# config = SimulationConfig(n_simulations=3, T=10, tsim=50, noise_percentage=0.1)
# simulator = CSTRSimulator(config)

# # # Run multiple simulations

# simulation_results = simulator.run_multiple_simulations()

# # Plot the results
# simulator.plot_results(simulation_results)