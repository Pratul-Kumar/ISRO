"""
Utility functions for raster processing, slope/aspect calculations, and file I/O operations.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
import rasterio.features
import rasterio.mask
from rasterio.crs import CRS
import os
import yaml
from typing import Tuple, List, Dict, Optional, Union
import logging
from pathlib import Path
# try:
#     from osgeo import gdal
# except ImportError:
#     import gdal
import geopandas as gpd
from scipy import ndimage
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RasterProcessor:
    """Utility class for raster processing operations."""
    
    def __init__(self, target_resolution: float = 30.0, target_crs: str = 'EPSG:4326'):
        """
        Initialize RasterProcessor.
        
        Args:
            target_resolution: Target resolution in meters
            target_crs: Target coordinate reference system
        """
        self.target_resolution = target_resolution
        self.target_crs = CRS.from_string(target_crs)
        
    def read_raster(self, file_path: str, band: int = 1) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Read raster file and return data array and profile.
        
        Args:
            file_path: Path to raster file
            band: Band number to read (1-indexed)
            
        Returns:
            Tuple of (data_array, profile)
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read(band)
                profile = src.profile.copy()
                return data, profile
        except Exception as e:
            logger.error(f"Error reading raster {file_path}: {e}")
            raise
            
    def write_raster(self, data: np.ndarray, profile: rasterio.profiles.Profile, 
                     output_path: str, compress: str = 'lzw') -> None:
        """
        Write raster data to file.
        
        Args:
            data: Data array to write
            profile: Raster profile
            output_path: Output file path
            compress: Compression method
        """
        try:
            # Update profile
            profile.update(
                dtype=data.dtype,
                height=data.shape[0],
                width=data.shape[1],
                compress=compress
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if len(data.shape) == 2:
                    dst.write(data, 1)
                else:
                    for i in range(data.shape[0]):
                        dst.write(data[i], i + 1)
                        
            logger.info(f"Raster saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error writing raster to {output_path}: {e}")
            raise
            
    def resample_raster(self, src_path: str, dst_path: str, 
                       target_width: int, target_height: int,
                       target_transform: rasterio.Affine,
                       resampling_method: Resampling = Resampling.bilinear) -> None:
        """
        Resample raster to target resolution and extent.
        
        Args:
            src_path: Source raster path
            dst_path: Destination raster path
            target_width: Target width in pixels
            target_height: Target height in pixels
            target_transform: Target geotransform
            resampling_method: Resampling method
        """
        try:
            with rasterio.open(src_path) as src:
                # Calculate target profile
                target_profile = src.profile.copy()
                target_profile.update({
                    'crs': self.target_crs,
                    'transform': target_transform,
                    'width': target_width,
                    'height': target_height
                })
                
                # Create output directory
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                with rasterio.open(dst_path, 'w', **target_profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=self.target_crs,
                            resampling=resampling_method
                        )
                        
            logger.info(f"Resampled raster saved to {dst_path}")
            
        except Exception as e:
            logger.error(f"Error resampling raster {src_path}: {e}")
            raise
            

def calculate_slope_aspect(dem: np.ndarray, pixel_size: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from DEM.
    
    Args:
        dem: Digital Elevation Model array
        pixel_size: Pixel size in meters
        
    Returns:
        Tuple of (slope_degrees, aspect_degrees)
    """
    try:
        # Handle nodata values
        dem = np.where(dem == -9999, np.nan, dem)
        
        # Check if array is large enough for gradient calculation
        if dem.shape[0] < 2 or dem.shape[1] < 2:
            logger.warning(f"DEM array too small for gradient calculation: {dem.shape}")
            # Return zero arrays with same shape
            return np.zeros_like(dem), np.zeros_like(dem)
        
        # Calculate gradients
        dy, dx = np.gradient(dem, pixel_size)
        
        # Calculate slope in radians then convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect in radians then convert to degrees
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert aspect to 0-360 range
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
        
        # Handle flat areas (slope = 0)
        aspect_deg = np.where(slope_deg == 0, -1, aspect_deg)
        
        return slope_deg, aspect_deg
        
    except Exception as e:
        logger.error(f"Error calculating slope and aspect: {e}")
        raise


def normalize_raster(data: np.ndarray, method: str = 'minmax', 
                    percentile_range: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Normalize raster data.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'percentile')
        percentile_range: Percentile range for percentile normalization
        
    Returns:
        Normalized data array
    """
    try:
        # Handle nodata values
        valid_mask = ~np.isnan(data) & (data != -9999)
        
        if method == 'minmax':
            min_val = np.nanmin(data[valid_mask])
            max_val = np.nanmax(data[valid_mask])
            # Avoid division by zero
            if max_val == min_val:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            mean_val = np.nanmean(data[valid_mask])
            std_val = np.nanstd(data[valid_mask])
            # Avoid division by zero
            if std_val == 0:
                normalized = np.zeros_like(data)
            else:
                normalized = (data - mean_val) / std_val
            
        elif method == 'percentile':
            p_min = np.nanpercentile(data[valid_mask], percentile_range[0])
            p_max = np.nanpercentile(data[valid_mask], percentile_range[1])
            # Avoid division by zero
            if p_max == p_min:
                normalized = np.zeros_like(data)
            else:
                normalized = np.clip((data - p_min) / (p_max - p_min), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        # Preserve nodata values
        normalized = np.where(valid_mask, normalized, np.nan)
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing raster: {e}")
        raise


def calculate_distance_to_roads(roads_shp: str, raster_template: str, 
                               output_path: str, max_distance: float = 5000.0) -> None:
    """
    Calculate distance to roads raster.
    
    Args:
        roads_shp: Path to roads shapefile
        raster_template: Template raster for output grid
        output_path: Output raster path
        max_distance: Maximum distance to calculate in meters
    """
    try:
        # Check if roads shapefile exists
        if not os.path.exists(roads_shp):
            logger.warning(f"Roads shapefile not found: {roads_shp}. Creating dummy distance raster.")
            # Create a dummy distance raster filled with max_distance
            with rasterio.open(raster_template) as template:
                dummy_distance = np.full(template.shape, max_distance, dtype=np.float32)
                
                profile = template.profile.copy()
                profile.update(dtype=np.float32, count=1)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(dummy_distance, 1)
                
                logger.info(f"Dummy distance raster saved to {output_path}")
            return
        
        # Read roads and template raster
        roads = gpd.read_file(roads_shp)
        
        with rasterio.open(raster_template) as template:
            # Rasterize roads
            shapes = ((geom, 1) for geom in roads.geometry)
            roads_raster = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=template.shape,
                transform=template.transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Calculate distance transform
            distance_raster = ndimage.distance_transform_edt(
                roads_raster == 0,
                sampling=[abs(template.transform[4]), abs(template.transform[0])]
            )
            
            # Clip to maximum distance
            distance_raster = np.clip(distance_raster, 0, max_distance)
            
            # Save result
            profile = template.profile.copy()
            profile.update(dtype=np.float32)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(distance_raster.astype(np.float32), 1)
                
        logger.info(f"Distance to roads raster saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error calculating distance to roads: {e}")
        raise


def encode_land_cover(lulc_data: np.ndarray, encoding_map: Dict[int, float]) -> np.ndarray:
    """
    Encode land use/land cover data to fuel load values.
    
    Args:
        lulc_data: Land use/land cover data array
        encoding_map: Mapping from LULC classes to fuel load values
        
    Returns:
        Encoded fuel load array
    """
    try:
        fuel_load = np.zeros_like(lulc_data, dtype=np.float32)
        
        for lulc_class, fuel_value in encoding_map.items():
            fuel_load[lulc_data == lulc_class] = fuel_value
            
        return fuel_load
        
    except Exception as e:
        logger.error(f"Error encoding land cover: {e}")
        raise


def stack_rasters(raster_paths: List[str], output_path: str) -> None:
    """
    Stack multiple rasters into a multi-band raster.
    
    Args:
        raster_paths: List of raster file paths
        output_path: Output multi-band raster path
    """
    try:
        # Read first raster to get template
        with rasterio.open(raster_paths[0]) as src:
            profile = src.profile.copy()
            profile.update(count=len(raster_paths))
            
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write stacked raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, raster_path in enumerate(raster_paths, 1):
                with rasterio.open(raster_path) as src:
                    data = src.read(1)
                    dst.write(data, i)
                    
        logger.info(f"Stacked raster saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error stacking rasters: {e}")
        raise


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def create_output_directories(base_path: str) -> Dict[str, str]:
    """
    Create output directory structure.
    
    Args:
        base_path: Base output path
        
    Returns:
        Dictionary of output directories
    """
    dirs = {
        'predictions': os.path.join(base_path, 'predictions'),
        'simulations': os.path.join(base_path, 'simulations'),
        'animations': os.path.join(base_path, 'animations'),
        'metrics': os.path.join(base_path, 'metrics'),
        'visualizations': os.path.join(base_path, 'visualizations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def validate_raster_alignment(raster_paths: List[str]) -> bool:
    """
    Validate that rasters have the same extent and resolution.
    
    Args:
        raster_paths: List of raster file paths
        
    Returns:
        True if aligned, False otherwise
    """
    try:
        reference = None
        
        for path in raster_paths:
            with rasterio.open(path) as src:
                if reference is None:
                    reference = {
                        'bounds': src.bounds,
                        'shape': src.shape,
                        'transform': src.transform,
                        'crs': src.crs
                    }
                else:
                    if (src.bounds != reference['bounds'] or
                        src.shape != reference['shape'] or
                        src.transform != reference['transform']):
                        return False
                        
        return True
        
    except Exception as e:
        logger.error(f"Error validating raster alignment: {e}")
        return False


def clip_rasters_to_extent(raster_paths: List[str], extent_shp: str, 
                          output_dir: str) -> List[str]:
    """
    Clip rasters to study area extent.
    
    Args:
        raster_paths: List of raster file paths
        extent_shp: Shapefile defining study area extent
        output_dir: Output directory for clipped rasters
        
    Returns:
        List of clipped raster paths
    """
    try:
        # Read extent shapefile
        extent_gdf = gpd.read_file(extent_shp)
        
        clipped_paths = []
        
        for raster_path in raster_paths:
            # Generate output path
            filename = os.path.basename(raster_path)
            name, ext = os.path.splitext(filename)
            clipped_path = os.path.join(output_dir, f"{name}_clipped{ext}")
            
            # Clip raster
            with rasterio.open(raster_path) as src:
                clipped_data, clipped_transform = rasterio.mask.mask(
                    src, extent_gdf.geometry, crop=True)
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2],
                    'transform': clipped_transform
                })
                
                # Save clipped raster
                with rasterio.open(clipped_path, 'w', **profile) as dst:
                    dst.write(clipped_data)
                    
            clipped_paths.append(clipped_path)
            
        logger.info(f"Clipped {len(raster_paths)} rasters to extent")
        return clipped_paths
        
    except Exception as e:
        logger.error(f"Error clipping rasters to extent: {e}")
        raise


# Suppress GDAL warnings
warnings.filterwarnings("ignore", category=UserWarning)
# gdal.UseExceptions()
