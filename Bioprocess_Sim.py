import numpy as np
from pcgym import make_env
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional, Union, Iterator
from dataclasses import dataclass
from scipy.stats import qmc
from tqdm import tqdm

@dataclass
class SimulationResult:
    """Container to store simulation results."""
    observed_states: Dict[str, List[float]]
    actions: Dict[str, List[float]]
    cons_check: Dict[bool, List[float]]
    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        """Makes SimulationResult iterable, yielding (observed_states, disturbances, actions)"""
        return iter((self.observed_states, self.actions, self.cons_check))

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int
    tsim: int
    noise_percentage: float = 0.01


class BioProcessSimulator:
    def __init__(
        self,
        config: SimulationConfig,
        constraints: Optional[Callable] = None
    ):
        """
        Initialize the BioProcess simulator with given parameters.
        
        Args:
            T (int): Number of time steps
            tsim (int): Simulation time period
            noise_percentage (float): Noise level for simulation
            constraints (Callable, optional): Constraint function that takes state and action as input
        """
        self.config = config
        self.constraints = constraints or (lambda x, u: np.array([x[1] - 800, x[2] - 0.011*x[0]]).reshape(-1,))
        
        # Define spaces
        self.action_space = {
            'low': np.array([120, 0]),
            'high': np.array([400, 40])
        }
        
        self.observation_space = {
            'low': np.array([0, 0, 0]),
            'high': np.array([10000, 10000, 10000])
        }
        
        self.uncertainty_space = {
            'low': np.array([0]*3),
            'high': np.array([1]*3)
        }
        
        self.uncertainties = {
            'x0': np.array([0.1, 0.1, 0.0]),
            'k_s': 0.1,
            'k_i': 0.10,
            'k_N': 0.10
        }
        
        self.x0 = np.array([1.0, 150.0, 0.0])

    def generate_lhs_actions(self) -> np.ndarray:
        """Generate 2D action sequence using Latin Hypercube Sampling with more frequent step changes."""
        # Create LHS sampler for 2D actions
        sampler = qmc.LatinHypercube(d=2)
        
        # Generate more base action values using LHS
        max_actions = self.config.T  # Increased from T//2 to T to allow more unique actions
        samples = sampler.random(n=max_actions)
        
        # Scale samples to action space - handling 2D actions
        action_range = self.action_space['high'] - self.action_space['low']
        action_samples = self.action_space['low'] + samples * action_range.reshape(1, -1)
        
        # Generate more change points for actions
        # Increase minimum number of changes and potential maximum
        min_changes = max_actions // 3  # Ensures at least 33% of timesteps have changes
        num_changes = np.random.randint(min_changes, max_actions)
        
        # Generate change points with minimum spacing
        min_spacing = 3  # Minimum timesteps between changes
        available_points = list(range(1, self.config.tsim))
        change_points = []
        
        while len(change_points) < num_changes and available_points:
            point = np.random.choice(available_points)
            change_points.append(point)
            
            # Remove nearby points to maintain minimum spacing
            for i in range(max(1, point - min_spacing), min(self.config.tsim, point + min_spacing + 1)):
                if i in available_points:
                    available_points.remove(i)
                    
        change_points = np.sort(change_points)
        
        # Initialize action sequence
        actions = []
        action_idx = 0
        current_action = action_samples[action_idx] + (np.random.rand(2) - 0.5) * 0.1  # Small variation
        
        # Generate action sequence with step changes
        for t in range(self.config.tsim):
            if len(change_points) > 0 and t == change_points[0]:
                action_idx = (action_idx + 1) % len(action_samples)
                current_action = action_samples[action_idx] + (np.random.rand(2) - 0.5) * 0.1
                change_points = change_points[1:]
            actions.append(current_action)
        
        return self._normalize_action(np.array(actions))

    def simulate(self) -> SimulationResult:
        """
        Run the simulation with LHS-generated actions and return the observed states and actions.
        
        Returns:
            SimulationResult: Formatted simulation results
        """
        observed_states = []
        uncertain_parameters = []
        actions = []
        const_values = []
        noiseless_observed_states = []
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': self.x0,
            'model': 'photo_production',
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'reward_states': np.array(['c_q']),
            'maximise_reward': True,
            'constraints': self.constraints,
            'uncertainty_bounds': self.uncertainty_space,
            'uncertainty_percentages': self.uncertainties,
            'distribution': 'normal',
            'done_on_cons_vio': False,
            'r_penalty': 1e6,
            'cons_type': "<=",
        }
        
        self.env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        # Generate complete action sequence using LHS
        action_sequence = self.generate_lhs_actions()
        
        obs, _ = self.env.reset()
        noiseless_obs, _ = noiseless_env.reset()
        done = False
        step = 0
        
        # Simulation loop
        while not done:
            # Get current action from LHS sequence
            action = action_sequence[step]
            
            # Run simulation step
            obs, _, done, _, info = self.env.step(action)
            noiseless_obs, _, _, _, _ = noiseless_env.step(action)
            con_check, g = self.env.constraint_check(obs, action)
            
            # Split and process observations
            uncertain_params = obs[3:]
            obs = obs[:3]
            
            noiseless_obs = noiseless_obs[:3]
            
            # Unnormalize values
            obs_unnorm = self._unnormalize_observation(obs)
            actions_unnorm = self._unnormalize_action(action)
            uncertain_params_unnorm = self._unnormalize_uncertainty(uncertain_params)
            
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs)
            
            # Store results
            observed_states.append(obs_unnorm)
            actions.append(actions_unnorm)
            uncertain_parameters.append(uncertain_params_unnorm)
            const_values.append(g)
            noiseless_observed_states.append(noiseless_obs_unnorm)
            
            # Increment step counter
            step += 1
        
        return self._format_results(observed_states, actions, const_values), self._format_results(noiseless_observed_states, actions, const_values)
    
    def run_multiple_simulations(self) -> List[SimulationResult]:
        """
        Run multiple simulations and return results.
        
        Args:
            num_simulations (int): Number of simulations to run
            
        Returns:
            List[SimulationResult]: List of simulation results
        """
        noisy_results = []
        noiseless_results = []
        
        for _ in tqdm(range(self.config.n_simulations), desc="Running simulations"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
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

    def _unnormalize_uncertainty(self, params: np.ndarray) -> np.ndarray:
        """Convert normalized uncertainty parameters back to original range."""
        return (params + 1) * (
            self.uncertainty_space['high'] - self.uncertainty_space['low']
        ) / 2 + self.uncertainty_space['low']

    def _format_results(
        self,
        observed_states: List[np.ndarray],
        actions: List[np.ndarray],
        con_check: List[np.ndarray]
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Format the simulation results into a structured dictionary."""
        obs_states = {
            'c_x': [state[0] for state in observed_states],
            'c_N': [state[1] for state in observed_states],
            'c_q': [state[2] for state in observed_states],
        }
        
        formatted_actions = {
            'I': [action[0] for action in actions],
            'F_N': [action[1] for action in actions]
        }
        

        con_check = {
            'constraint_1': [check[0] for check in con_check],
            'constraint_2': [check[1] for check in con_check]
        }
        
        return SimulationResult(obs_states, formatted_actions, con_check)

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        """
        Plot results from multiple simulations, including both noisy and noiseless results.
        
        Args:
            noisy_results (List[SimulationResult]): List of noisy simulation results
            noiseless_results (List[SimulationResult]): List of noiseless simulation results
        """
        # Create subplots for observed states
        fig_obs, axs_obs = plt.subplots(3, 1, figsize=(15, 9))
        fig_act, axs_act = plt.subplots(2, 1, figsize=(12, 8))
        fig_con, axs_con = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(noisy_results)))
        
        # Plot observed states
        for i, ((noisy_result, noiseless_result), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            axs_obs[0].plot(noisy_result.observed_states['c_x'], label=f'Noisy Simulation {i+1}', color = color)
            axs_obs[0].plot(noiseless_result.observed_states['c_x'], linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_obs[0].set_title('Concentration of Biomass (g/L)')
            axs_obs[0].set_xlabel('Time (h)')
            axs_obs[0].set_ylabel('c_X')
            
            axs_obs[1].plot(noisy_result.observed_states['c_N'], label=f'Noisy Simulation {i+1}', color = color)
            axs_obs[1].plot(noiseless_result.observed_states['c_N'], linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_obs[1].set_title('Nitrate Concentration (g/L)')
            axs_obs[1].axhline(y=800, color='r', linestyle='--', label='Constraint: c_N < 800')
            axs_obs[1].set_xlabel('Time (h)')
            axs_obs[1].set_ylabel('c_N')
            
            axs_obs[2].plot(noisy_result.observed_states['c_q'], label=f'Noisy Simulation {i+1}', color = color)
            axs_obs[2].plot(noiseless_result.observed_states['c_q'], linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_obs[2].set_title('Bioproduct Concentration (g/L)')
            axs_obs[2].set_xlabel('Time (h)')
            axs_obs[2].set_ylabel('c_q')
        
        # Plot actions
        for i, ((noisy_result, noiseless_result), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            axs_act[0].plot(noisy_result.actions['I'], label=f'Noisy Simulation {i+1}', color = color)
            axs_act[0].plot(noiseless_result.actions['I'], linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_act[0].set_title('Light Intensity (micromol / m^2 s)')
            axs_act[0].set_xlabel('Time (h)')
            axs_act[0].set_ylabel('I')
        
            axs_act[1].plot(noisy_result.actions['F_N'], label=f'Noisy Simulation {i+1}', color = color)
            axs_act[1].plot(noiseless_result.actions['F_N'], linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_act[1].set_title('Nitrate Flowrate (mg / L h)')
            axs_act[1].set_xlabel('Time (h)')
            axs_act[1].set_ylabel('F_N')
        
        for i, ((noisy_result, noiseless_result), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            axs_con.plot(np.array(noisy_result.observed_states['c_q']) / np.array(noisy_result.observed_states['c_x']), label=f'Noisy Simulation {i+1}', color = color)
            axs_con.plot(np.array(noiseless_result.observed_states['c_q']) / np.array(noiseless_result.observed_states['c_x']), linestyle='--', label=f'Noiseless Simulation {i+1}', color = color)
            axs_con.axhline(y=0.011, color='r', linestyle='--')
            axs_con.set_title('Ratio of Bioproduct to Biomass Concentration Constraint')
            axs_con.set_xlabel('Time (h)')
            axs_con.set_ylabel('c_q/c_x')
            
        # Add legends
        for ax in axs_obs:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        for ax in axs_act:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        axs_con.legend(loc='upper right', fontsize='small', ncol=2)
        # Adjust layouts
        for fig in [fig_obs, fig_act, fig_con]:
            fig.tight_layout()
        plt.show()

        
# sim_config = SimulationConfig(n_simulations=10, T=20, tsim=240, noise_percentage=0.01)
# simulator = BioProcessSimulator(sim_config)
# simulation_results, noiseless_results = simulator.run_multiple_simulations()
# simulator.plot_results(simulation_results, noiseless_results)
