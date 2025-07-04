"""
Fire spread simulation using Cellular Automata (CA).
Implements dynamic fire spread modeling based on weather, terrain, and fuel conditions.
"""

import numpy as np
import rasterio
from rasterio.profiles import default_gtiff_profile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import imageio
import os
import logging
from typing import List, Dict, Tuple, Optional
from scipy import ndimage
import cv2
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class CellularAutomataFireSpread:
    """Cellular Automata model for fire spread simulation."""
    
    def __init__(self, config: Dict):
        """
        Initialize CA fire spread model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Model parameters
        self.cell_size = config.get('cell_size', 30)  # meters
        self.time_step = config.get('time_step', 300)  # seconds (5 minutes)
        self.max_burning_time = config.get('max_burning_time', 7200)  # seconds (2 hours)
        
        # Fire spread parameters
        self.base_spread_rate = config.get('base_spread_rate', 0.1)  # m/s
        self.wind_effect = config.get('wind_effect', 0.5)
        self.slope_effect = config.get('slope_effect', 0.3)
        self.fuel_effect = config.get('fuel_effect', 0.8)
        self.moisture_effect = config.get('moisture_effect', 0.6)
        
        # Fire states
        self.UNBURNED = 0
        self.BURNING = 1
        self.BURNED = 2
        self.NO_FUEL = 3
        
        # Initialize state arrays
        self.fire_state = None
        self.burning_time = None
        self.fuel_load = None
        self.moisture_content = None
        self.wind_speed = None
        self.wind_direction = None
        self.slope = None
        self.aspect = None
        
    def load_environmental_data(self, data_paths: Dict[str, str]):
        """
        Load environmental data for fire spread simulation.
        
        Args:
            data_paths: Dictionary of environmental data file paths
        """
        logger.info("Loading environmental data...")
        
        # Load fuel load
        if 'fuel_load' in data_paths:
            with rasterio.open(data_paths['fuel_load']) as src:
                self.fuel_load = src.read(1).astype(np.float32)
                self.profile = src.profile.copy()
                
        # Load moisture content
        if 'moisture' in data_paths:
            with rasterio.open(data_paths['moisture']) as src:
                self.moisture_content = src.read(1).astype(np.float32)
        else:
            # Default moisture content if not provided
            self.moisture_content = np.full_like(self.fuel_load, 0.3)
            
        # Load wind data
        if 'wind_speed' in data_paths:
            with rasterio.open(data_paths['wind_speed']) as src:
                self.wind_speed = src.read(1).astype(np.float32)
        else:
            self.wind_speed = np.full_like(self.fuel_load, 5.0)  # Default 5 m/s
            
        if 'wind_direction' in data_paths:
            with rasterio.open(data_paths['wind_direction']) as src:
                self.wind_direction = src.read(1).astype(np.float32)
        else:
            self.wind_direction = np.full_like(self.fuel_load, 45.0)  # Default NE
            
        # Load terrain data
        if 'slope' in data_paths:
            with rasterio.open(data_paths['slope']) as src:
                self.slope = src.read(1).astype(np.float32)
        else:
            self.slope = np.zeros_like(self.fuel_load)
            
        if 'aspect' in data_paths:
            with rasterio.open(data_paths['aspect']) as src:
                self.aspect = src.read(1).astype(np.float32)
        else:
            self.aspect = np.zeros_like(self.fuel_load)
            
        # Initialize fire state
        self.fire_state = np.zeros_like(self.fuel_load, dtype=np.int32)
        self.burning_time = np.zeros_like(self.fuel_load, dtype=np.float32)
        
        # Set no-fuel areas
        self.fire_state[self.fuel_load <= 0] = self.NO_FUEL
        
        logger.info(f"Environmental data loaded: {self.fire_state.shape}")
        
    def initialize_ignition_points(self, ignition_data: np.ndarray):
        """
        Initialize fire ignition points.
        
        Args:
            ignition_data: Binary array indicating ignition locations
        """
        ignition_mask = ignition_data > 0
        self.fire_state[ignition_mask] = self.BURNING
        self.burning_time[ignition_mask] = 0
        
        logger.info(f"Initialized {np.sum(ignition_mask)} ignition points")
        
    def calculate_spread_probability(self, i: int, j: int, ni: int, nj: int) -> float:
        """
        Calculate fire spread probability from cell (i,j) to neighbor (ni,nj).
        
        Args:
            i, j: Source cell coordinates
            ni, nj: Neighbor cell coordinates
            
        Returns:
            Spread probability [0-1]
        """
        # Base spread rate
        prob = self.base_spread_rate * self.time_step / self.cell_size
        
        # Fuel effect
        if self.fuel_load[ni, nj] > 0:
            fuel_factor = 1 + self.fuel_effect * self.fuel_load[ni, nj]
        else:
            return 0.0  # No fuel, no spread
            
        # Moisture effect (higher moisture reduces spread)
        moisture_factor = 1 - self.moisture_effect * self.moisture_content[ni, nj]
        moisture_factor = max(0.1, moisture_factor)  # Minimum factor
        
        # Wind effect
        wind_factor = self._calculate_wind_effect(i, j, ni, nj)
        
        # Slope effect
        slope_factor = self._calculate_slope_effect(i, j, ni, nj)
        
        # Combined probability
        prob *= fuel_factor * moisture_factor * wind_factor * slope_factor
        
        # Distance factor (diagonal vs adjacent)
        if abs(ni - i) + abs(nj - j) == 2:  # Diagonal
            prob *= 0.707  # sqrt(2)/2
            
        return min(1.0, prob)
        
    def _calculate_wind_effect(self, i: int, j: int, ni: int, nj: int) -> float:
        """Calculate wind effect on fire spread."""
        # Direction from source to neighbor
        di, dj = ni - i, nj - j
        spread_angle = np.degrees(np.arctan2(dj, di))
        
        # Wind direction (meteorological convention: direction wind comes from)
        wind_dir = self.wind_direction[i, j]
        wind_speed = self.wind_speed[i, j]
        
        # Angle difference between wind and spread direction
        angle_diff = abs(spread_angle - (wind_dir + 180)) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        # Wind effect factor
        wind_alignment = np.cos(np.radians(angle_diff))
        wind_factor = 1 + self.wind_effect * wind_speed * wind_alignment / 10.0
        
        return max(0.1, wind_factor)
        
    def _calculate_slope_effect(self, i: int, j: int, ni: int, nj: int) -> float:
        """Calculate slope effect on fire spread."""
        # Elevation difference (approximate)
        slope_i = self.slope[i, j]
        slope_ni = self.slope[ni, nj]
        
        # Direction from source to neighbor
        di, dj = ni - i, nj - j
        spread_angle = np.degrees(np.arctan2(dj, di))
        
        # Aspect at source cell
        aspect_i = self.aspect[i, j]
        
        # Slope alignment with spread direction
        slope_alignment = np.cos(np.radians(spread_angle - aspect_i))
        
        # Upslope enhances spread, downslope reduces it
        avg_slope = (slope_i + slope_ni) / 2
        slope_factor = 1 + self.slope_effect * avg_slope * slope_alignment / 45.0
        
        return max(0.1, slope_factor)
        
    def update_fire_state(self):
        """Update fire state for one time step."""
        new_fire_state = self.fire_state.copy()
        new_burning_time = self.burning_time.copy()
        
        # Find currently burning cells
        burning_cells = np.where(self.fire_state == self.BURNING)
        
        for i, j in zip(burning_cells[0], burning_cells[1]):
            # Update burning time
            new_burning_time[i, j] += self.time_step
            
            # Check if cell burns out
            if new_burning_time[i, j] >= self.max_burning_time:
                new_fire_state[i, j] = self.BURNED
                continue
                
            # Spread to neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                        
                    ni, nj = i + di, j + dj
                    
                    # Check bounds
                    if (ni < 0 or ni >= self.fire_state.shape[0] or
                        nj < 0 or nj >= self.fire_state.shape[1]):
                        continue
                        
                    # Check if neighbor can burn
                    if self.fire_state[ni, nj] != self.UNBURNED:
                        continue
                        
                    # Calculate spread probability
                    spread_prob = self.calculate_spread_probability(i, j, ni, nj)
                    
                    # Stochastic ignition
                    if np.random.random() < spread_prob:
                        new_fire_state[ni, nj] = self.BURNING
                        new_burning_time[ni, nj] = 0
                        
        self.fire_state = new_fire_state
        self.burning_time = new_burning_time
        
    def simulate_fire_spread(self, ignition_points: np.ndarray,
                           simulation_hours: float = 12.0,
                           save_interval_hours: float = 1.0) -> List[np.ndarray]:
        """
        Run fire spread simulation.
        
        Args:
            ignition_points: Initial ignition locations
            simulation_hours: Total simulation time in hours
            save_interval_hours: Interval to save states in hours
            
        Returns:
            List of fire state arrays at save intervals
        """
        logger.info(f"Starting fire spread simulation for {simulation_hours} hours...")
        
        # Initialize
        self.initialize_ignition_points(ignition_points)
        
        # Simulation parameters
        total_steps = int(simulation_hours * 3600 / self.time_step)
        save_interval_steps = int(save_interval_hours * 3600 / self.time_step)
        
        # Store simulation states
        simulation_states = [self.fire_state.copy()]
        time_stamps = [0]
        
        # Run simulation
        for step in range(total_steps):
            self.update_fire_state()
            
            # Save state at intervals
            if (step + 1) % save_interval_steps == 0:
                simulation_states.append(self.fire_state.copy())
                elapsed_hours = (step + 1) * self.time_step / 3600
                time_stamps.append(elapsed_hours)
                
                # Log progress
                burning_cells = np.sum(self.fire_state == self.BURNING)
                burned_cells = np.sum(self.fire_state == self.BURNED)
                logger.info(f"Hour {elapsed_hours:.1f}: {burning_cells} burning, {burned_cells} burned")
                
        logger.info("Fire spread simulation completed")
        
        return simulation_states, time_stamps
        
    def save_simulation_results(self, simulation_states: List[np.ndarray],
                              time_stamps: List[float],
                              output_dir: str) -> List[str]:
        """
        Save simulation results as GeoTIFF files.
        
        Args:
            simulation_states: List of fire state arrays
            time_stamps: List of time stamps
            output_dir: Output directory
            
        Returns:
            List of output file paths
        """
        logger.info("Saving simulation results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        
        for i, (state, timestamp) in enumerate(zip(simulation_states, time_stamps)):
            # Create filename
            filename = f"fire_spread_hour_{timestamp:04.1f}.tif"
            output_path = os.path.join(output_dir, filename)
            
            # Update profile
            profile = self.profile.copy()
            profile.update(dtype=np.int32, count=1)
            
            # Save GeoTIFF
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(state, 1)
                
            output_paths.append(output_path)
            
        logger.info(f"Saved {len(output_paths)} simulation results")
        return output_paths
        
    def create_fire_animation(self, simulation_states: List[np.ndarray],
                            time_stamps: List[float],
                            output_path: str,
                            fps: int = 2) -> str:
        """
        Create animated visualization of fire spread.
        
        Args:
            simulation_states: List of fire state arrays
            time_stamps: List of time stamps
            output_path: Output animation file path
            fps: Frames per second
            
        Returns:
            Path to animation file
        """
        logger.info("Creating fire spread animation...")
        
        # Create colormap
        colors = ['lightgreen', 'red', 'black', 'gray']
        cmap = ListedColormap(colors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame):
            ax.clear()
            state = simulation_states[frame]
            timestamp = time_stamps[frame]
            
            im = ax.imshow(state, cmap=cmap, vmin=0, vmax=3)
            ax.set_title(f'Fire Spread Simulation - Hour {timestamp:.1f}')
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Unburned'),
                plt.Rectangle((0,0),1,1, facecolor='red', label='Burning'),
                plt.Rectangle((0,0),1,1, facecolor='black', label='Burned'),
                plt.Rectangle((0,0),1,1, facecolor='gray', label='No Fuel')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            return [im]
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(simulation_states),
            interval=1000//fps, blit=False, repeat=True)
        
        # Save animation
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='FireSim'), bitrate=1800)
            anim.save(output_path, writer=writer)
        elif output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
            
        plt.close(fig)
        logger.info(f"Animation saved to {output_path}")
        
        return output_path
        
    def calculate_fire_statistics(self, simulation_states: List[np.ndarray],
                                time_stamps: List[float]) -> Dict:
        """
        Calculate fire spread statistics.
        
        Args:
            simulation_states: List of fire state arrays
            time_stamps: List of time stamps
            
        Returns:
            Dictionary of fire statistics
        """
        stats = {
            'time_stamps': time_stamps,
            'burning_area': [],
            'burned_area': [],
            'total_affected_area': [],
            'spread_rate': [],
            'fire_perimeter': []
        }
        
        cell_area = (self.cell_size ** 2) / 10000  # Convert to hectares
        
        for i, state in enumerate(simulation_states):
            burning_cells = np.sum(state == self.BURNING)
            burned_cells = np.sum(state == self.BURNED)
            total_affected = burning_cells + burned_cells
            
            stats['burning_area'].append(burning_cells * cell_area)
            stats['burned_area'].append(burned_cells * cell_area)
            stats['total_affected_area'].append(total_affected * cell_area)
            
            # Calculate spread rate
            if i > 0:
                area_increase = stats['total_affected_area'][i] - stats['total_affected_area'][i-1]
                time_diff = time_stamps[i] - time_stamps[i-1]
                spread_rate = area_increase / time_diff if time_diff > 0 else 0
                stats['spread_rate'].append(spread_rate)
            else:
                stats['spread_rate'].append(0)
                
            # Calculate fire perimeter
            fire_mask = (state == self.BURNING) | (state == self.BURNED)
            perimeter = self._calculate_perimeter(fire_mask) * self.cell_size / 1000  # km
            stats['fire_perimeter'].append(perimeter)
            
        return stats
        
    def _calculate_perimeter(self, fire_mask: np.ndarray) -> int:
        """Calculate fire perimeter in number of cell edges."""
        # Use morphological operations to find perimeter
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(fire_mask.astype(np.uint8), kernel, iterations=1)
        perimeter_mask = dilated - fire_mask.astype(np.uint8)
        return np.sum(perimeter_mask)
        
    def save_statistics(self, stats: Dict, output_path: str):
        """Save fire statistics to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info(f"Statistics saved to {output_path}")


class FireSpreadAnalyzer:
    """Analysis tools for fire spread simulation results."""
    
    def __init__(self):
        pass
        
    def analyze_spread_patterns(self, simulation_states: List[np.ndarray],
                              environmental_data: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze fire spread patterns in relation to environmental factors.
        
        Args:
            simulation_states: List of fire state arrays
            environmental_data: Dictionary of environmental arrays
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Analyzing fire spread patterns...")
        
        # Get final burned area
        final_state = simulation_states[-1]
        burned_mask = (final_state == 2)  # BURNED
        
        analysis = {}
        
        # Analyze by fuel load
        if 'fuel_load' in environmental_data:
            fuel_data = environmental_data['fuel_load']
            analysis['fuel_analysis'] = self._analyze_by_factor(burned_mask, fuel_data, 'fuel_load')
            
        # Analyze by slope
        if 'slope' in environmental_data:
            slope_data = environmental_data['slope']
            analysis['slope_analysis'] = self._analyze_by_factor(burned_mask, slope_data, 'slope')
            
        # Analyze by wind speed
        if 'wind_speed' in environmental_data:
            wind_data = environmental_data['wind_speed']
            analysis['wind_analysis'] = self._analyze_by_factor(burned_mask, wind_data, 'wind_speed')
            
        return analysis
        
    def _analyze_by_factor(self, burned_mask: np.ndarray, 
                          factor_data: np.ndarray, factor_name: str) -> Dict:
        """Analyze burned area by environmental factor."""
        burned_values = factor_data[burned_mask]
        unburned_values = factor_data[~burned_mask]
        
        # Remove NaN values
        burned_values = burned_values[~np.isnan(burned_values)]
        unburned_values = unburned_values[~np.isnan(unburned_values)]
        
        analysis = {
            'burned_mean': float(np.mean(burned_values)) if len(burned_values) > 0 else 0,
            'burned_std': float(np.std(burned_values)) if len(burned_values) > 0 else 0,
            'unburned_mean': float(np.mean(unburned_values)) if len(unburned_values) > 0 else 0,
            'unburned_std': float(np.std(unburned_values)) if len(unburned_values) > 0 else 0,
            'factor_name': factor_name
        }
        
        return analysis
        
    def plot_spread_analysis(self, stats: Dict, output_path: str):
        """Plot fire spread analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time_stamps = stats['time_stamps']
        
        # Total affected area over time
        axes[0, 0].plot(time_stamps, stats['total_affected_area'], 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Area (hectares)')
        axes[0, 0].set_title('Total Affected Area Over Time')
        axes[0, 0].grid(True)
        
        # Burning vs burned area
        axes[0, 1].plot(time_stamps, stats['burning_area'], 'r-', label='Burning', linewidth=2)
        axes[0, 1].plot(time_stamps, stats['burned_area'], 'k-', label='Burned', linewidth=2)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Area (hectares)')
        axes[0, 1].set_title('Burning vs Burned Area')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Spread rate
        axes[1, 0].plot(time_stamps[1:], stats['spread_rate'][1:], 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Spread Rate (ha/hour)')
        axes[1, 0].set_title('Fire Spread Rate')
        axes[1, 0].grid(True)
        
        # Fire perimeter
        axes[1, 1].plot(time_stamps, stats['fire_perimeter'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Perimeter (km)')
        axes[1, 1].set_title('Fire Perimeter Length')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis plots saved to {output_path}")


def run_fire_spread_simulation(config: Dict, ignition_raster_path: str) -> Dict:
    """
    Run complete fire spread simulation pipeline.
    
    Args:
        config: Configuration dictionary
        ignition_raster_path: Path to ignition points raster
        
    Returns:
        Dictionary containing simulation results
    """
    logger.info("Running fire spread simulation pipeline...")
    
    # Initialize simulation
    ca_model = CellularAutomataFireSpread(config)
    
    # Load environmental data
    env_data_paths = config['environmental_data_paths']
    ca_model.load_environmental_data(env_data_paths)
    
    # Load ignition points
    with rasterio.open(ignition_raster_path) as src:
        ignition_points = src.read(1)
        
    # Run simulation
    simulation_hours = config.get('simulation_hours', 12.0)
    save_interval = config.get('save_interval_hours', 1.0)
    
    simulation_states, time_stamps = ca_model.simulate_fire_spread(
        ignition_points, simulation_hours, save_interval)
    
    # Save results
    output_dir = config['paths']['outputs']
    simulation_dir = os.path.join(output_dir, 'simulations')
    
    # Save GeoTIFF files
    geotiff_paths = ca_model.save_simulation_results(
        simulation_states, time_stamps, simulation_dir)
    
    # Create animation
    animation_path = os.path.join(output_dir, 'animations', 'fire_spread.mp4')
    ca_model.create_fire_animation(simulation_states, time_stamps, animation_path)
    
    # Calculate statistics
    stats = ca_model.calculate_fire_statistics(simulation_states, time_stamps)
    stats_path = os.path.join(output_dir, 'metrics', 'fire_spread_stats.json')
    ca_model.save_statistics(stats, stats_path)
    
    # Create analysis
    analyzer = FireSpreadAnalyzer()
    
    # Load environmental data for analysis
    env_data = {}
    for key, path in env_data_paths.items():
        with rasterio.open(path) as src:
            env_data[key] = src.read(1)
            
    spread_analysis = analyzer.analyze_spread_patterns(simulation_states, env_data)
    
    # Plot analysis
    plots_path = os.path.join(output_dir, 'visualizations', 'spread_analysis.png')
    analyzer.plot_spread_analysis(stats, plots_path)
    
    logger.info("Fire spread simulation pipeline completed")
    
    return {
        'simulation_states': simulation_states,
        'time_stamps': time_stamps,
        'statistics': stats,
        'analysis': spread_analysis,
        'output_paths': {
            'geotiffs': geotiff_paths,
            'animation': animation_path,
            'statistics': stats_path,
            'analysis_plot': plots_path
        }
    }
