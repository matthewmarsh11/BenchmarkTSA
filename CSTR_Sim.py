import numpy as np
from pcgym import make_env
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass
from tqdm import tqdm

np.random.seed(42)

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
        num_changes = np.random.randint(0, self.config.T // 2)
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
        disturbance_space = {'low': np.array([320, 0.7]), 'high': np.array([350, 0.8])}
        
        nominal_temp = np.random.uniform(disturbance_space['low'][0], disturbance_space['high'][0])
        nominal_conc = np.random.uniform(disturbance_space['low'][1], disturbance_space['high'][1])
        
        # Temperature disturbances
        num_changes_temp = np.random.randint(0, self.config.T // 2)
        change_points_temp = np.sort(np.random.choice(range(1, self.config.tsim), num_changes_temp, replace=False))
        disturbances_temp = []
        current_disturbance_temp = nominal_temp + (np.random.rand() - 0.5) * 10
        
        for t in range(self.config.tsim):
            if len(change_points_temp) > 0 and t == change_points_temp[0]:
                current_disturbance_temp = nominal_temp + (np.random.rand() - 0.5) * 10
                change_points_temp = change_points_temp[1:]
            disturbances_temp.append(current_disturbance_temp)
        
        # Concentration disturbances
        num_changes_conc = np.random.randint(0, self.config.T // 2)
        change_points_conc = np.sort(np.random.choice(range(1, self.config.tsim), num_changes_conc, replace=False))
        disturbances_conc = []
        current_disturbance_conc = nominal_conc + (np.random.rand() - 0.5) * 0.1
        
        for t in range(self.config.tsim):
            if len(change_points_conc) > 0 and t == change_points_conc[0]:
                current_disturbance_conc = nominal_conc + (np.random.rand() - 0.5) * 0.1
                change_points_conc = change_points_conc[1:]
            disturbances_conc.append(current_disturbance_conc)
        
        disturbances = {'Ti': disturbances_temp, 'Caf': disturbances_conc}
  
        
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
        # Create a noiseless environment for comparison
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        
        env = make_env(env_params)
        noiseless_env = make_env(noiseless_env_params)
        
        # Create lists to store observed states, disturbances, and actions
        observed_states = []
        disturbance_values = []
        actions = []
        obs, _ = env.reset()
        done = False
        
        
        noiseless_observed_states = []
        noiseless_disturbance_values = []
        obs, _ = noiseless_env.reset()
        
        num_actions = np.random.randint(0, self.config.T)
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_actions, replace=False))
        initial_action = np.random.uniform(self.action_space['low'], self.action_space['high'])
        initial_action = self._normalize_action(initial_action)
        # Simulation loop
        while not done:
            if not actions:
                action = initial_action
            elif num_actions > 0 and env.t == change_points[0]:
                if np.random.rand() < 0.5:
                    action = np.random.uniform(self.action_space['low'], self.action_space['high'])
                    action = self._normalize_action(action)
                    change_points = change_points[1:]
                    num_actions -= 1
                else:
                    action = self._normalize_action(actions[-1])
            
            obs, _, done, _, info = env.step(action)
            noiseless_obs, _, _, _, _ = noiseless_env.step(action)
            
            disturbance = obs[3:5]
            uncertain_params = obs[5:]
            obs = obs[:3]
            
            noiseless_disturbance = noiseless_obs[3:5]
            noiseless_uncertain_params = noiseless_obs[5:]
            noiseless_obs = noiseless_obs[:3]
            
            obs_unnorm = self._unnormalize_observation(obs)
            disturbance_unnorm = self._unnormalize_disturbance(disturbance, disturbance_space)
            actions_unnorm = self._unnormalize_action(action)
            
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs)
            noiseless_disturbance_unnorm = self._unnormalize_disturbance(noiseless_disturbance, disturbance_space)
            
            observed_states.append(obs_unnorm)
            disturbance_values.append(disturbance_unnorm)
            
            noiseless_observed_states.append(noiseless_obs_unnorm)
            noiseless_disturbance_values.append(noiseless_disturbance_unnorm)
            
            actions.append(actions_unnorm)
        
        return self._format_results(observed_states, disturbance_values, actions), self._format_results(noiseless_observed_states, noiseless_disturbance_values, actions)
        
    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        """
        Run multiple simulations and return both noisy and noiseless results.
        
        Returns:
            Tuple[List[SimulationResult], List[SimulationResult]]: Lists of (noisy_results, noiseless_results)
        """
        noisy_results = []
        noiseless_results = []
        
        for _ in tqdm(range(self.config.n_simulations), desc="Running simulations"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results
    
    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        """
        Plot all simulation results:
        - States (Ca, T) with both noisy and noiseless data
        - Actions (Tc) from noisy data only
        - Disturbances (Ti, Caf) from noisy data only
        
        Args:
            noisy_results (List[SimulationResult]): List of noisy simulation results
            noiseless_results (List[SimulationResult]): List of noiseless simulation results
        """
        # Create three separate figures for states, actions, and disturbances
        fig_states, axs_states = plt.subplots(2, 1, figsize=(10, 8))
        fig_act, ax_act = plt.subplots(1, 1, figsize=(10, 4))
        fig_dist, axs_dist = plt.subplots(2, 1, figsize=(10, 8))
        
        # Get a color for each simulation pair
        colors = plt.cm.tab10(np.linspace(0, 1, len(noisy_results)))
        
        # Plot states (with noisy/noiseless comparison)
        for i, ((noisy, noiseless), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            # Concentration - Ca
            axs_states[0].plot(noisy.observed_states['Ca'], 
                            label=f'Simulation {i+1}', 
                            color=color,
                            alpha=0.7)
            axs_states[0].plot(noiseless.observed_states['Ca'], 
                            label=f'Noiseless {i+1}', 
                            color=color,
                            linestyle='--', 
                            alpha=0.7)
            
            # Temperature - T
            axs_states[1].plot(noisy.observed_states['T'], 
                            label=f'Simulation {i+1}', 
                            color=color,
                            alpha=0.7)
            axs_states[1].plot(noiseless.observed_states['T'], 
                            label=f'Noiseless {i+1}', 
                            color=color,
                            linestyle='--', 
                            alpha=0.7)
            
            # Plot actions (noisy only)
            ax_act.plot(noisy.actions['Tc'], 
                    label=f'Simulation {i+1}', 
                    color=color,
                    alpha=0.7)
            
            # Plot disturbances (noisy only)
            axs_dist[0].plot(noisy.disturbances['Ti'], 
                            label=f'Simulation {i+1}', 
                            color=color,
                            alpha=0.7)
            axs_dist[1].plot(noisy.disturbances['Caf'], 
                            label=f'Simulation {i+1}', 
                            color=color,
                            alpha=0.7)
        
        # Set titles and labels for states
        axs_states[0].set_title('Concentration of A (Ca)')
        axs_states[0].set_xlabel('Time')
        axs_states[0].set_ylabel('Ca')
        axs_states[1].set_title('Temperature (T)')
        axs_states[1].set_xlabel('Time')
        axs_states[1].set_ylabel('T')
        
        # Set titles and labels for actions
        ax_act.set_title('Coolant Temperature (Tc)')
        ax_act.set_xlabel('Time')
        ax_act.set_ylabel('Tc')
        
        # Set titles and labels for disturbances
        axs_dist[0].set_title('Inlet Temperature (Ti)')
        axs_dist[0].set_xlabel('Time')
        axs_dist[0].set_ylabel('Ti')
        axs_dist[1].set_title('Feed Concentration (Caf)')
        axs_dist[1].set_xlabel('Time')
        axs_dist[1].set_ylabel('Caf')
        
        # Add legends to all plots
        for ax in axs_states:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax_act.legend(loc='upper right', fontsize='small', ncol=2)
        for ax in axs_dist:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        
        # Adjust layouts
        for fig in [fig_states, fig_act, fig_dist]:
            fig.tight_layout()
        
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