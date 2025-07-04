"""
Data preprocessing module for forest fire prediction.
Handles loading, aligning, normalizing, and stacking raster layers.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import geopandas as gpd
from utils import (
    RasterProcessor, calculate_slope_aspect, normalize_raster,
    calculate_distance_to_roads, encode_land_cover, stack_rasters,
    validate_raster_alignment, clip_rasters_to_extent
)

logger = logging.getLogger(__name__)


class FireDataPreprocessor:
    """Data preprocessing pipeline for forest fire prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_resolution = config.get('target_resolution', 30.0)
        self.target_crs = config.get('target_crs', 'EPSG:4326')
        self.processor = RasterProcessor(self.target_resolution, self.target_crs)
        
        # LULC to fuel load mapping
        self.fuel_encoding = {
            1: 0.1,   # Urban/Built-up
            2: 0.3,   # Agricultural
            3: 0.8,   # Forest/Woody
            4: 0.6,   # Grassland
            5: 0.9,   # Shrubland
            6: 0.0,   # Water
            7: 0.2,   # Barren
            8: 0.4,   # Wetland
        }
        
    def load_and_align_rasters(self, data_paths: Dict[str, str], 
                              extent_bounds: Optional[Tuple] = None) -> Dict[str, str]:
        """
        Load and align all raster datasets to common grid.
        
        Args:
            data_paths: Dictionary mapping data type to file path
            extent_bounds: Optional bounding box (minx, miny, maxx, maxy)
            
        Returns:
            Dictionary of aligned raster paths
        """
        logger.info("Starting raster alignment process...")
        
        # Create processed data directory
        processed_dir = self.config['paths']['processed_data']
        os.makedirs(processed_dir, exist_ok=True)
        
        aligned_paths = {}
        
        # Determine target grid from DEM or first raster
        reference_path = data_paths.get('dem') or list(data_paths.values())[0]
        target_transform, target_width, target_height = self._calculate_target_grid(
            reference_path, extent_bounds)
        
        # Process each raster (excluding fire_history which is processed separately)
        for data_type, file_path in data_paths.items():
            if data_type == 'fire_history':
                continue  # Skip fire history - processed separately later
                
            logger.info(f"Processing {data_type}: {file_path}")
            
            aligned_path = os.path.join(processed_dir, f"{data_type}_aligned.tif")
            
            if data_type == 'dem':
                # Process DEM and derive slope/aspect
                aligned_path = self._process_dem(file_path, aligned_path, 
                                               target_transform, target_width, target_height)
            elif data_type == 'lulc':
                # Process land use/land cover
                aligned_path = self._process_lulc(file_path, aligned_path,
                                                target_transform, target_width, target_height)
            else:
                # Process other rasters (weather, etc.)
                aligned_path = self._align_raster(file_path, aligned_path,
                                                target_transform, target_width, target_height)
            
            aligned_paths[data_type] = aligned_path
            
        logger.info("Raster alignment completed")
        return aligned_paths
        
    def _calculate_target_grid(self, reference_path: str, 
                              extent_bounds: Optional[Tuple] = None) -> Tuple:
        """Calculate target grid parameters."""
        with rasterio.open(reference_path) as src:
            if extent_bounds:
                # Use provided bounds
                transform, width, height = calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height,
                    *extent_bounds, resolution=self.target_resolution)
            else:
                # Use full extent of reference raster
                transform, width, height = calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height,
                    *src.bounds, resolution=self.target_resolution)
                
        return transform, width, height
        
    def _align_raster(self, src_path: str, dst_path: str, 
                     target_transform: rasterio.Affine, 
                     target_width: int, target_height: int) -> str:
        """Align raster to target grid."""
        self.processor.resample_raster(
            src_path, dst_path, target_width, target_height,
            target_transform, Resampling.bilinear)
        return dst_path
        
    def _process_dem(self, dem_path: str, output_path: str,
                    target_transform: rasterio.Affine,
                    target_width: int, target_height: int) -> str:
        """Process DEM and derive terrain parameters."""
        logger.info("Processing DEM and deriving slope/aspect...")
        
        # Align DEM first
        aligned_dem_path = output_path.replace('.tif', '_dem.tif')
        self._align_raster(dem_path, aligned_dem_path, target_transform, 
                          target_width, target_height)
        
        # Read aligned DEM
        dem_data, profile = self.processor.read_raster(aligned_dem_path)
        
        # Calculate slope and aspect
        slope, aspect = calculate_slope_aspect(dem_data, self.target_resolution)
        
        # Save slope
        slope_path = output_path.replace('.tif', '_slope.tif')
        self.processor.write_raster(slope, profile, slope_path)
        
        # Save aspect
        aspect_path = output_path.replace('.tif', '_aspect.tif')
        self.processor.write_raster(aspect, profile, aspect_path)
        
        # Return DEM path (we'll handle slope/aspect separately)
        return aligned_dem_path
        
    def _process_lulc(self, lulc_path: str, output_path: str,
                     target_transform: rasterio.Affine,
                     target_width: int, target_height: int) -> str:
        """Process land use/land cover data."""
        logger.info("Processing LULC data...")
        
        # Align LULC using nearest neighbor (preserve classes)
        aligned_lulc_path = output_path.replace('.tif', '_lulc.tif')
        self.processor.resample_raster(
            lulc_path, aligned_lulc_path, target_width, target_height,
            target_transform, Resampling.nearest)
        
        # Read aligned LULC
        lulc_data, profile = self.processor.read_raster(aligned_lulc_path)
        
        # Encode to fuel load
        fuel_load = encode_land_cover(lulc_data, self.fuel_encoding)
        
        # Save fuel load
        fuel_path = output_path.replace('.tif', '_fuel.tif')
        self.processor.write_raster(fuel_load, profile, fuel_path)
        
        return aligned_lulc_path
        
    def normalize_weather_data(self, weather_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Normalize weather data.
        
        Args:
            weather_paths: Dictionary of weather variable paths
            
        Returns:
            Dictionary of normalized weather paths
        """
        logger.info("Normalizing weather data...")
        
        normalized_paths = {}
        
        for var_name, file_path in weather_paths.items():
            # Read data
            data, profile = self.processor.read_raster(file_path)
            
            # Apply appropriate normalization
            if var_name in ['temperature', 'humidity']:
                normalized_data = normalize_raster(data, method='minmax')
            elif var_name in ['wind_speed', 'rainfall']:
                normalized_data = normalize_raster(data, method='percentile')
            else:
                normalized_data = normalize_raster(data, method='zscore')
            
            # Save normalized data
            output_path = file_path.replace('.tif', '_normalized.tif')
            self.processor.write_raster(normalized_data, profile, output_path)
            normalized_paths[var_name] = output_path
            
        return normalized_paths
        
    def create_feature_stack(self, feature_paths: Dict[str, str], 
                           output_path: str) -> str:
        """
        Create multi-band feature stack.
        
        Args:
            feature_paths: Dictionary of feature raster paths
            output_path: Output stacked raster path
            
        Returns:
            Path to stacked raster
        """
        logger.info("Creating feature stack...")
        
        # Define feature order for consistent stacking
        feature_order = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'rainfall', 'slope', 'aspect', 'fuel_load', 'elevation',
            'distance_to_roads', 'distance_to_settlements'
        ]
        
        # Collect available features in order
        ordered_paths = []
        for feature in feature_order:
            if feature in feature_paths:
                ordered_paths.append(feature_paths[feature])
                
        # Add any additional features not in the standard order
        for feature, path in feature_paths.items():
            if feature not in feature_order:
                ordered_paths.append(path)
                
        # Stack rasters
        stack_rasters(ordered_paths, output_path)
        
        logger.info(f"Feature stack created with {len(ordered_paths)} bands")
        return output_path
        
    def prepare_training_data(self, feature_stack_path: str, 
                            fire_history_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data arrays.
        
        Args:
            feature_stack_path: Path to feature stack
            fire_history_path: Path to fire history raster
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Preparing training data...")
        
        # Read feature stack
        with rasterio.open(feature_stack_path) as src:
            features = src.read()  # Shape: (bands, height, width)
            feature_profile = src.profile
            
        # Read fire history (labels)
        with rasterio.open(fire_history_path) as src:
            labels = src.read(1)  # Shape: (height, width)
            label_profile = src.profile
            
        logger.info(f"Feature stack shape: {features.shape}")
        logger.info(f"Fire history shape: {labels.shape}")
        
        # Check if grids match
        if features.shape[1:] != labels.shape:
            logger.error(f"Grid mismatch: features {features.shape[1:]} vs labels {labels.shape}")
            # Resample fire history to match feature stack
            from utils import RasterProcessor
            processor = RasterProcessor(self.target_resolution, self.target_crs)
            
            # Create temporary aligned fire history using nearest neighbor for binary data
            temp_fire_path = fire_history_path.replace('.tif', '_aligned_temp.tif')
            processor.resample_raster(
                fire_history_path, temp_fire_path,
                features.shape[2], features.shape[1],  # width, height
                feature_profile['transform'],
                resampling_method=Resampling.nearest  # Use nearest neighbor for binary data
            )
            
            # Re-read aligned fire history
            with rasterio.open(temp_fire_path) as src:
                labels = src.read(1)
            
            # Clean up temp file
            import os
            os.remove(temp_fire_path)
            
        # Reshape features: (bands, height, width) -> (height*width, bands)
        n_bands, height, width = features.shape
        features = features.reshape(n_bands, -1).T
        
        # Reshape labels: (height, width) -> (height*width,)
        labels = labels.reshape(-1)
        
        logger.info(f"Reshaped features: {features.shape}")
        logger.info(f"Reshaped labels: {labels.shape}")
        
        # Remove nodata pixels
        valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels) & (labels != -9999)
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Convert labels to binary (fire/no fire)
        labels = (labels > 0).astype(np.int32)
        
        logger.info(f"Training data prepared: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Fire pixels: {np.sum(labels)}, No-fire pixels: {np.sum(1-labels)}")
        
        return features, labels
        
    def create_prediction_grid(self, feature_stack_path: str) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Create prediction grid from feature stack.
        
        Args:
            feature_stack_path: Path to feature stack
            
        Returns:
            Tuple of (features_grid, profile)
        """
        with rasterio.open(feature_stack_path) as src:
            features = src.read()
            profile = src.profile.copy()
            
        # Reshape for prediction: (bands, height, width) -> (height*width, bands)
        n_bands, height, width = features.shape
        features_grid = features.reshape(n_bands, -1).T
        
        # Handle nodata values
        valid_mask = ~np.isnan(features_grid).any(axis=1)
        
        return features_grid, profile, valid_mask, (height, width)
        
    def process_historical_fires(self, fire_data_paths: List[str], 
                                output_path: str) -> str:
        """
        Process historical fire occurrence data.
        
        Args:
            fire_data_paths: List of fire occurrence raster paths
            output_path: Output composite fire history path
            
        Returns:
            Path to processed fire history raster
        """
        logger.info("Processing historical fire data...")
        
        if not fire_data_paths:
            raise ValueError("No fire data paths provided")
            
        # Read first fire raster as template
        with rasterio.open(fire_data_paths[0]) as src:
            fire_composite = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            
        # Aggregate multiple fire datasets
        for fire_path in fire_data_paths[1:]:
            with rasterio.open(fire_path) as src:
                fire_data = src.read(1)
                # Add to composite (could be sum, max, or other aggregation)
                fire_composite += fire_data
                
        # Normalize to 0-1 range
        fire_composite = fire_composite / len(fire_data_paths)
        
        # Save processed fire history
        profile.update(dtype=np.float32)
        self.processor.write_raster(fire_composite, profile, output_path)
        
        return output_path
        
    def run_preprocessing_pipeline(self, config_data: Dict) -> Dict[str, str]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            config_data: Configuration with data paths
            
        Returns:
            Dictionary of processed data paths
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Extract paths from config
        raw_data_paths = config_data['data_paths']
        
        # Step 1: Load and align rasters
        aligned_paths = self.load_and_align_rasters(raw_data_paths)
        
        # Step 2: Process DEM derivatives (slope, aspect)
        if 'dem' in aligned_paths:
            dem_path = aligned_paths['dem']
            base_path = dem_path.replace('_aligned.tif', '')
            
            slope_path = f"{base_path}_slope.tif"
            aspect_path = f"{base_path}_aspect.tif"
            
            if os.path.exists(slope_path):
                aligned_paths['slope'] = slope_path
            if os.path.exists(aspect_path):
                aligned_paths['aspect'] = aspect_path
                
        # Step 3: Process LULC to fuel load
        if 'lulc' in aligned_paths:
            lulc_path = aligned_paths['lulc']
            fuel_path = lulc_path.replace('_lulc.tif', '_fuel.tif')
            if os.path.exists(fuel_path):
                aligned_paths['fuel_load'] = fuel_path
                
        # Step 4: Normalize weather data
        weather_vars = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'rainfall']
        weather_paths = {var: aligned_paths[var] for var in weather_vars if var in aligned_paths}
        
        if weather_paths:
            normalized_weather = self.normalize_weather_data(weather_paths)
            aligned_paths.update(normalized_weather)
            
        # Step 5: Calculate distance rasters (if shape files provided)
        if 'roads_shapefile' in config_data:
            roads_distance_path = os.path.join(
                self.config['paths']['processed_data'], 'distance_to_roads.tif')
            calculate_distance_to_roads(
                config_data['roads_shapefile'],
                list(aligned_paths.values())[0],  # Use any aligned raster as template
                roads_distance_path
            )
            aligned_paths['distance_to_roads'] = roads_distance_path
            
        # Step 6: Create feature stack
        feature_stack_path = os.path.join(
            self.config['paths']['processed_data'], 'feature_stack.tif')
        
        # Filter paths for feature stack (exclude raw LULC, include fuel_load)
        feature_paths = {k: v for k, v in aligned_paths.items() 
                        if k not in ['lulc'] or k == 'fuel_load'}
        
        self.create_feature_stack(feature_paths, feature_stack_path)
        aligned_paths['feature_stack'] = feature_stack_path
        
        # Step 7: Process historical fire data
        if 'fire_history' in raw_data_paths:
            fire_history_paths = raw_data_paths['fire_history']
            if isinstance(fire_history_paths, str):
                fire_history_paths = [fire_history_paths]
                
            processed_fire_path = os.path.join(
                self.config['paths']['processed_data'], 'fire_history_processed.tif')
            
            self.process_historical_fires(fire_history_paths, processed_fire_path)
            aligned_paths['fire_history'] = processed_fire_path
            
        logger.info("Preprocessing pipeline completed successfully")
        
        return aligned_paths
