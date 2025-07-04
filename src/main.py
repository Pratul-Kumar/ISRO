"""
Main execution script for Forest Fire Prediction and Spread Simulation pipeline.
Orchestrates data preprocessing, model training, prediction, and fire spread simulation.
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
import warnings

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from utils import load_config, create_output_directories
from data_preprocessing import FireDataPreprocessor
from fire_prediction_model import create_fire_prediction_pipeline
from fire_spread_simulation import run_fire_spread_simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fire_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_environment(config_path: str) -> dict:
    """
    Setup environment and load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info("Setting up environment...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    base_output_path = config['paths']['outputs']
    output_dirs = create_output_directories(base_output_path)
    config['output_dirs'] = output_dirs
    
    # Set random seeds for reproducibility
    np.random.seed(config.get('random_seed', 42))
    
    logger.info("Environment setup completed")
    return config


def validate_input_data(config: dict, skip_preprocessing: bool = False) -> bool:
    """
    Validate that required input data files exist.
    
    Args:
        config: Configuration dictionary
        skip_preprocessing: If True, only validate raw data files
        
    Returns:
        True if all required files exist, False otherwise
    """
    logger.info("Validating input data...")
    
    required_files = []
    
    # Check raw data paths (always required)
    if 'data_paths' in config:
        for data_type, file_path in config['data_paths'].items():
            if isinstance(file_path, list):
                required_files.extend(file_path)
            else:
                required_files.append(file_path)
                
    # Check environmental data paths for simulation (only if not preprocessing)
    if skip_preprocessing and 'environmental_data_paths' in config:
        for data_type, file_path in config['environmental_data_paths'].items():
            required_files.append(file_path)
            
    # Validate file existence
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
        
    logger.info("Input data validation completed successfully")
    return True


def run_data_preprocessing(config: dict) -> dict:
    """
    Run data preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of processed data paths
    """
    logger.info("=" * 50)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 50)
    
    # Initialize preprocessor
    preprocessor = FireDataPreprocessor(config)
    
    # Run preprocessing pipeline
    processed_paths = preprocessor.run_preprocessing_pipeline(config)
    
    logger.info("Data preprocessing completed successfully")
    return processed_paths


def run_fire_prediction(config: dict, processed_paths: dict) -> dict:
    """
    Run fire prediction model training and evaluation.
    
    Args:
        config: Configuration dictionary
        processed_paths: Dictionary of processed data paths
        
    Returns:
        Dictionary containing model results
    """
    logger.info("=" * 50)
    logger.info("STARTING FIRE PREDICTION MODELING")
    logger.info("=" * 50)
    
    # Update config with processed paths
    config['processed_paths'] = processed_paths
    
    # Run prediction pipeline
    prediction_results = create_fire_prediction_pipeline(config)
    
    logger.info("Fire prediction modeling completed successfully")
    return prediction_results


def generate_fire_risk_map(config: dict, prediction_results: dict) -> str:
    """
    Generate fire risk prediction map for tomorrow.
    
    Args:
        config: Configuration dictionary
        prediction_results: Model prediction results
        
    Returns:
        Path to generated risk map
    """
    logger.info("Generating fire risk prediction map...")
    
    # Load current conditions for prediction
    processed_paths = config['processed_paths']
    
    from data_preprocessing import FireDataPreprocessor
    preprocessor = FireDataPreprocessor(config)
    
    # Create prediction grid
    features_grid, profile, valid_mask, grid_shape = preprocessor.create_prediction_grid(
        processed_paths['feature_stack'])
    
    # Make predictions
    model = prediction_results['model']
    
    # Handle different model types
    if config['model_type'] == 'unet':
        # Reshape for U-Net prediction
        height, width = grid_shape
        n_features = features_grid.shape[1]
        
        # Create image patches for prediction
        patch_size = config.get('patch_size', (256, 256))
        features_image = features_grid.reshape(height, width, n_features)
        
        # For simplicity, predict on the whole image (might need tiling for large images)
        if height <= patch_size[0] and width <= patch_size[1]:
            # Pad if necessary
            pad_h = max(0, patch_size[0] - height)
            pad_w = max(0, patch_size[1] - width)
            
            features_padded = np.pad(features_image, 
                                   ((0, pad_h), (0, pad_w), (0, 0)), 
                                   mode='constant', constant_values=0)
            
            # Predict
            prediction = model.predict(np.expand_dims(features_padded, 0))[0]
            
            # Remove padding
            prediction = prediction[:height, :width, 0]
        else:
            # For large images, would need tiling approach
            logger.warning("Image too large for current U-Net implementation")
            prediction = np.zeros((height, width))
            
    else:  # LSTM or other models
        # Standard prediction
        predictions = model.predict(features_grid[valid_mask])
        
        # Reshape to grid
        prediction = np.zeros(grid_shape[0] * grid_shape[1])
        prediction[valid_mask] = predictions.flatten()
        prediction = prediction.reshape(grid_shape)
    
    # Save prediction map
    output_path = os.path.join(
        config['output_dirs']['predictions'], 
        f"fire_risk_map_{datetime.now().strftime('%Y%m%d')}.tif"
    )
    
    # Update profile for output
    profile.update(dtype=np.float32, count=1)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(np.float32), 1)
        
    logger.info(f"Fire risk map saved to {output_path}")
    return output_path


def run_fire_spread_simulation_pipeline(config: dict, fire_risk_map_path: str) -> dict:
    """
    Run fire spread simulation using predicted high-risk areas as ignition points.
    
    Args:
        config: Configuration dictionary
        fire_risk_map_path: Path to fire risk prediction map
        
    Returns:
        Dictionary containing simulation results
    """
    logger.info("=" * 50)
    logger.info("STARTING FIRE SPREAD SIMULATION")
    logger.info("=" * 50)
    
    # Create ignition points from high-risk areas
    with rasterio.open(fire_risk_map_path) as src:
        risk_map = src.read(1)
        profile = src.profile.copy()
        
    # Define high-risk threshold
    risk_threshold = config.get('risk_threshold', 0.8)
    ignition_points = (risk_map >= risk_threshold).astype(np.int32)
    
    # Save ignition points
    ignition_path = os.path.join(
        config['output_dirs']['predictions'], 
        "ignition_points.tif"
    )
    
    with rasterio.open(ignition_path, 'w', **profile) as dst:
        dst.write(ignition_points, 1)
        
    logger.info(f"Created {np.sum(ignition_points)} ignition points")
    
    # Run simulation
    simulation_results = run_fire_spread_simulation(config, ignition_path)
    
    logger.info("Fire spread simulation completed successfully")
    return simulation_results


def create_summary_report(config: dict, prediction_results: dict, 
                         simulation_results: dict) -> str:
    """
    Create summary report of the entire pipeline run.
    
    Args:
        config: Configuration dictionary
        prediction_results: Model prediction results
        simulation_results: Simulation results
        
    Returns:
        Path to summary report
    """
    logger.info("Creating summary report...")
    
    report_path = os.path.join(
        config['output_dirs']['metrics'],
        f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    with open(report_path, 'w') as f:
        f.write("FOREST FIRE PREDICTION AND SIMULATION SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {config['model_type']}\n\n")
        
        # Model performance
        if 'metrics' in prediction_results:
            metrics = prediction_results['metrics']
            f.write("MODEL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            if 'roc_auc' in metrics:
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write("\n")
            
        # Simulation statistics
        if 'statistics' in simulation_results:
            stats = simulation_results['statistics']
            f.write("FIRE SPREAD SIMULATION STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total simulation time: {max(stats['time_stamps']):.1f} hours\n")
            f.write(f"Maximum affected area: {max(stats['total_affected_area']):.2f} hectares\n")
            f.write(f"Maximum spread rate: {max(stats['spread_rate']):.2f} ha/hour\n")
            f.write(f"Final fire perimeter: {stats['fire_perimeter'][-1]:.2f} km\n")
            f.write("\n")
            
        # Output files
        f.write("OUTPUT FILES:\n")
        f.write("-" * 15 + "\n")
        
        if 'output_paths' in simulation_results:
            paths = simulation_results['output_paths']
            f.write(f"Animation: {paths['animation']}\n")
            f.write(f"Statistics: {paths['statistics']}\n")
            f.write(f"Analysis Plot: {paths['analysis_plot']}\n")
            f.write(f"Simulation GeoTIFFs: {len(paths['geotiffs'])} files\n")
            
    logger.info(f"Summary report saved to {report_path}")
    return report_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Forest Fire Prediction and Spread Simulation Pipeline"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip data preprocessing step'
    )
    parser.add_argument(
        '--skip-prediction', 
        action='store_true',
        help='Skip fire prediction modeling step'
    )
    parser.add_argument(
        '--skip-simulation', 
        action='store_true',
        help='Skip fire spread simulation step'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Forest Fire Prediction and Simulation Pipeline")
        logger.info("=" * 60)
        
        # Setup environment
        config = setup_environment(args.config)
        
        # Validate input data
        if not validate_input_data(config, args.skip_preprocessing):
            logger.error("Input data validation failed. Exiting.")
            sys.exit(1)
            
        processed_paths = None
        prediction_results = None
        fire_risk_map_path = None
        simulation_results = None
        
        # Run preprocessing
        if not args.skip_preprocessing:
            processed_paths = run_data_preprocessing(config)
        else:
            logger.info("Skipping data preprocessing...")
            
        # Run fire prediction
        if not args.skip_prediction and processed_paths:
            prediction_results = run_fire_prediction(config, processed_paths)
            
            # Generate fire risk map
            fire_risk_map_path = generate_fire_risk_map(config, prediction_results)
        else:
            logger.info("Skipping fire prediction modeling...")
            
        # Run fire spread simulation
        if not args.skip_simulation and fire_risk_map_path:
            simulation_results = run_fire_spread_simulation_pipeline(
                config, fire_risk_map_path)
        else:
            logger.info("Skipping fire spread simulation...")
            
        # Create summary report
        if prediction_results and simulation_results:
            summary_path = create_summary_report(
                config, prediction_results, simulation_results)
                
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Print summary
        print("\n" + "=" * 60)
        print("FOREST FIRE PREDICTION PIPELINE COMPLETED")
        print("=" * 60)
        
        if prediction_results and 'metrics' in prediction_results:
            metrics = prediction_results['metrics']
            print(f"\nModel Performance:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            
        if simulation_results and 'statistics' in simulation_results:
            stats = simulation_results['statistics']
            print(f"\nSimulation Results:")
            print(f"  Total affected area: {max(stats['total_affected_area']):.2f} hectares")
            print(f"  Maximum spread rate: {max(stats['spread_rate']):.2f} ha/hour")
            
        print(f"\nOutput directory: {config['paths']['outputs']}")
        print("\nCheck the output directory for detailed results, maps, and animations.")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
