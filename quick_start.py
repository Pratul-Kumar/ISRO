#!/usr/bin/env python3
"""
Quick start script for Forest Fire Prediction and Simulation Pipeline.
This script sets up sample data and runs a demonstration of the complete pipeline.
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(data_dir: str, grid_size: tuple = (500, 500), 
                      bounds: tuple = (-120.5, 39.0, -120.0, 39.5)):
    """
    Create sample environmental data for demonstration.
    
    Args:
        data_dir: Directory to save sample data
        grid_size: Size of the raster grid (height, width)
        bounds: Geographic bounds (minx, miny, maxx, maxy)
    """
    logger.info("Creating sample environmental data...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    height, width = grid_size
    minx, miny, maxx, maxy = bounds
    
    # Create geotransform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crs = CRS.from_epsg(4326)
    
    # Base profile for all rasters
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': np.float32,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Temperature (°C) - varies with elevation and latitude
    elevation_effect = 100 + 1000 * np.sin(y_coords/100) * np.cos(x_coords/150)
    latitude_effect = 5 * (y_coords / height)
    temperature = 25 + latitude_effect - elevation_effect/200 + 3 * np.random.random((height, width))
    
    with rasterio.open(os.path.join(data_dir, 'temperature.tif'), 'w', **profile) as dst:
        dst.write(temperature.astype(np.float32), 1)
    
    # 2. Humidity (%) - inversely related to temperature
    humidity = 70 - 0.5 * temperature + 15 * np.random.random((height, width))
    humidity = np.clip(humidity, 0, 100)
    
    with rasterio.open(os.path.join(data_dir, 'humidity.tif'), 'w', **profile) as dst:
        dst.write(humidity.astype(np.float32), 1)
    
    # 3. Wind speed (m/s)
    wind_speed = 3 + 7 * np.random.random((height, width))
    
    with rasterio.open(os.path.join(data_dir, 'wind_speed.tif'), 'w', **profile) as dst:
        dst.write(wind_speed.astype(np.float32), 1)
    
    # 4. Wind direction (degrees)
    wind_direction = 360 * np.random.random((height, width))
    
    with rasterio.open(os.path.join(data_dir, 'wind_direction.tif'), 'w', **profile) as dst:
        dst.write(wind_direction.astype(np.float32), 1)
    
    # 5. Rainfall (mm)
    rainfall = 2 * np.random.exponential(1, (height, width))
    
    with rasterio.open(os.path.join(data_dir, 'rainfall.tif'), 'w', **profile) as dst:
        dst.write(rainfall.astype(np.float32), 1)
    
    # 6. Digital Elevation Model (m)
    dem = 500 + elevation_effect + 100 * np.random.random((height, width))
    
    with rasterio.open(os.path.join(data_dir, 'dem.tif'), 'w', **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
    
    # 7. Land Cover (classes: 1=urban, 2=agricultural, 3=forest, 4=grassland, 5=shrubland, 6=water, 7=barren)
    # Create realistic land cover patterns
    landcover = np.ones((height, width), dtype=np.int32) * 4  # Default to grassland
    
    # Add forests (higher elevation)
    forest_mask = dem > np.percentile(dem, 60)
    landcover[forest_mask] = 3
    
    # Add water bodies
    water_centers = [(height//4, width//3), (3*height//4, 2*width//3)]
    for center_y, center_x in water_centers:
        y_dist = (y_coords - center_y) ** 2
        x_dist = (x_coords - center_x) ** 2
        distance = np.sqrt(y_dist + x_dist)
        water_mask = distance < 20
        landcover[water_mask] = 6
    
    # Add urban areas
    urban_centers = [(height//2, width//4), (height//3, 3*width//4)]
    for center_y, center_x in urban_centers:
        y_dist = (y_coords - center_y) ** 2
        x_dist = (x_coords - center_x) ** 2
        distance = np.sqrt(y_dist + x_dist)
        urban_mask = distance < 15
        landcover[urban_mask] = 1
    
    # Add agricultural areas near urban
    for center_y, center_x in urban_centers:
        y_dist = (y_coords - center_y) ** 2
        x_dist = (x_coords - center_x) ** 2
        distance = np.sqrt(y_dist + x_dist)
        ag_mask = (distance >= 15) & (distance < 30)
        landcover[ag_mask] = 2
    
    # Add shrubland in transition zones
    shrub_mask = (dem > np.percentile(dem, 40)) & (dem <= np.percentile(dem, 60))
    landcover[shrub_mask] = 5
    
    # Add some barren areas at high elevation
    barren_mask = dem > np.percentile(dem, 90)
    landcover[barren_mask] = 7
    
    profile_int = profile.copy()
    profile_int['dtype'] = np.int32
    
    with rasterio.open(os.path.join(data_dir, 'landcover.tif'), 'w', **profile_int) as dst:
        dst.write(landcover, 1)
    
    # 8. Historical fire data (3 years)
    for year in [2020, 2021, 2022]:
        # Create fire probability based on conditions
        fire_prob = (
            0.02 * (temperature - 20) +  # Higher temp increases fire risk
            -0.01 * humidity +            # Lower humidity increases risk
            0.005 * wind_speed +          # Higher wind increases risk
            -0.5 * (landcover == 6) +     # Water bodies can't burn
            -0.3 * (landcover == 1) +     # Urban areas less likely
            0.3 * (landcover == 3) +      # Forests more likely
            0.2 * (landcover == 5)        # Shrublands more likely
        )
        
        # Add randomness and ensure positive probabilities
        fire_prob = np.maximum(0, fire_prob + 0.02 * np.random.random((height, width)))
        
        # Convert to binary fire occurrence (top 2% become fires)
        fire_threshold = np.percentile(fire_prob, 98)
        fire_occurrence = (fire_prob >= fire_threshold).astype(np.int32)
        
        # Add some clustering to make fires more realistic
        from scipy import ndimage
        fire_occurrence = ndimage.binary_dilation(fire_occurrence, iterations=1).astype(np.int32)
        
        with rasterio.open(os.path.join(data_dir, f'fire_{year}.tif'), 'w', **profile_int) as dst:
            dst.write(fire_occurrence, 1)
    
    logger.info(f"Sample data created in {data_dir}")
    logger.info("Files created:")
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            logger.info(f"  - {file}")


def update_config_paths(config_path: str, data_dir: str):
    """Update configuration file with correct data paths."""
    import yaml
    
    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data paths
    config['data_paths'] = {
        'temperature': os.path.join(data_dir, 'temperature.tif'),
        'humidity': os.path.join(data_dir, 'humidity.tif'),
        'wind_speed': os.path.join(data_dir, 'wind_speed.tif'),
        'wind_direction': os.path.join(data_dir, 'wind_direction.tif'),
        'rainfall': os.path.join(data_dir, 'rainfall.tif'),
        'dem': os.path.join(data_dir, 'dem.tif'),
        'lulc': os.path.join(data_dir, 'landcover.tif'),
        'fire_history': [
            os.path.join(data_dir, 'fire_2020.tif'),
            os.path.join(data_dir, 'fire_2021.tif'),
            os.path.join(data_dir, 'fire_2022.tif')
        ]
    }
    
    # Update other settings for quick demo
    config['epochs'] = 10  # Reduced for quick demo
    config['batch_size'] = 8  # Set to requested batch size
    config['simulation_hours'] = 6.0
    config['save_interval_hours'] = 1.0
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration updated: {config_path}")


def run_quick_demo():
    """Run a quick demonstration of the fire prediction pipeline."""
    logger.info("Starting Quick Demo of Forest Fire Prediction Pipeline")
    logger.info("=" * 60)
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    config_path = os.path.join(base_dir, 'config', 'default_config.yaml')
    
    # Create sample data
    create_sample_data(data_dir)
    
    # Update configuration
    update_config_paths(config_path, data_dir)
    
    # Run the main pipeline
    logger.info("Running main pipeline...")
    
    try:
        from src.main import main
        import sys
        
        # Set up arguments for main function
        sys.argv = ['main.py', '--config', config_path]
        
        # Run pipeline
        main()
        
        logger.info("Quick demo completed successfully!")
        logger.info("Check the data/outputs directory for results.")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        logger.info("You can run the pipeline manually with:")
        logger.info(f"python src/main.py --config {config_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick start for Forest Fire Prediction Pipeline")
    parser.add_argument('--create-data-only', action='store_true', 
                       help='Only create sample data without running pipeline')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory to create sample data')
    
    args = parser.parse_args()
    
    if args.create_data_only:
        # Just create sample data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, args.data_dir)
        create_sample_data(data_dir)
        
        # Update config
        config_path = os.path.join(base_dir, 'config', 'default_config.yaml')
        update_config_paths(config_path, data_dir)
        
        print(f"Sample data created in: {data_dir}")
        print(f"Configuration updated: {config_path}")
        print("You can now run: python src/main.py --config config/default_config.yaml")
    else:
        # Run full demo
        run_quick_demo()
