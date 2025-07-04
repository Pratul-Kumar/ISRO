# Forest Fire Prediction System - Technical Documentation

## 📋 Complete Dependencies & Import Guide

This document provides detailed information about all dependencies, imports, and their specific purposes in the Forest Fire Prediction and Spread Simulation pipeline.

## 🐍 Python Version Requirements

```python
Python >= 3.8, <= 3.12 (Recommended: Python 3.11)
```

**Why this version range?**
- TensorFlow 2.x requires Python 3.8+
- Python 3.12 has compatibility issues with some geospatial libraries
- Python 3.11 provides optimal performance and compatibility

## 📦 Core Dependencies & Imports

### 1. Machine Learning & Deep Learning

#### TensorFlow 2.x
```python
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
```

**Purpose:**
- Primary deep learning framework for building U-Net and LSTM models
- U-Net for spatial fire risk prediction
- LSTM for temporal pattern recognition in weather data
- GPU acceleration support for faster training

**Why TensorFlow?**
- Excellent support for convolutional neural networks (CNNs)
- Built-in support for image/raster data processing
- Comprehensive model training utilities
- Strong community support for geospatial applications

#### PyTorch (Alternative/Complementary)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
```

**Purpose:**
- Alternative deep learning framework
- More flexible for custom model architectures
- Research-oriented with dynamic computation graphs

### 2. Geospatial Data Processing

#### Rasterio
```python
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.mask import mask
```

**Purpose:**
- Primary library for reading/writing raster data (GeoTIFF, NetCDF)
- Handles coordinate reference systems (CRS) transformations
- Raster reprojection and resampling to 30m resolution
- Geometric operations on raster data

**Why Rasterio?**
- Industry standard for Python raster operations
- Excellent performance with large datasets
- GDAL backend for broad format support

#### GDAL/OGR
```python
from osgeo import gdal, osr, ogr
```

**Purpose:**
- Low-level geospatial data abstraction library
- Format translation and processing
- Coordinate system transformations
- Vector-raster conversions

#### GeoPandas
```python
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
```

**Purpose:**
- Vector data processing (shapefiles, GeoJSON)
- Road networks and settlement data handling
- Spatial joins and geometric operations
- Converting vector data to raster format

### 3. Scientific Computing & Numerical Operations

#### NumPy
```python
import numpy as np
```

**Purpose:**
- Foundation for all numerical computations
- Array operations for raster data manipulation
- Mathematical functions for fire spread calculations
- Memory-efficient data structures

#### SciPy
```python
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
```

**Purpose:**
- Advanced scientific computing functions
- Image processing operations (filtering, morphology)
- Spatial interpolation for missing data
- Distance calculations for proximity analysis

#### Pandas
```python
import pandas as pd
```

**Purpose:**
- Tabular data manipulation
- Time series data handling for weather parameters
- Data cleaning and preprocessing
- Statistical analysis and aggregation

### 4. Visualization & Plotting

#### Matplotlib
```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
```

**Purpose:**
- Static plotting and visualization
- Creating fire spread animations
- Model performance visualization
- Custom map plotting

#### Seaborn
```python
import seaborn as sns
```

**Purpose:**
- Statistical data visualization
- Correlation matrices and heatmaps
- Enhanced plotting aesthetics

#### Plotly
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
```

**Purpose:**
- Interactive visualizations
- Web-based fire risk maps
- 3D terrain visualization
- Dashboard creation

### 5. Image Processing

#### OpenCV
```python
import cv2
```

**Purpose:**
- Advanced image processing operations
- Morphological operations on fire spread patterns
- Contour detection for fire boundaries
- Video generation for fire spread animations

#### Scikit-Image
```python
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
```

**Purpose:**
- Scientific image analysis
- Feature extraction from satellite imagery
- Image segmentation for land cover classification
- Noise reduction and filtering

### 6. Machine Learning Utilities

#### Scikit-Learn
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
```

**Purpose:**
- Data preprocessing and normalization
- Train-test data splitting
- Model evaluation metrics
- Baseline machine learning models for comparison

### 7. Data Formats & I/O

#### NetCDF4
```python
import netCDF4 as nc
from netCDF4 import Dataset
```

**Purpose:**
- Reading weather data (temperature, humidity, wind)
- Climate model outputs
- Multi-dimensional array handling
- Metadata preservation

#### HDF5
```python
import h5py
```

**Purpose:**
- Hierarchical data format for large datasets
- Efficient storage and retrieval
- Satellite data processing
- Model weight storage

### 8. System & Utility Libraries

#### OS & System
```python
import os
import sys
import glob
from pathlib import Path
```

**Purpose:**
- File system operations
- Path management across platforms
- Directory creation and navigation
- File pattern matching

#### Configuration & Logging
```python
import yaml
import json
import logging
from datetime import datetime, timedelta
```

**Purpose:**
- Configuration file management
- Structured logging for debugging
- Date-time operations for temporal data
- JSON data handling

#### Multiprocessing
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
```

**Purpose:**
- Parallel processing for large raster datasets
- Speeding up data preprocessing
- Concurrent model training
- Efficient CPU utilization

## 🔧 Installation Commands

### Complete Installation
```bash
# Create virtual environment
python -m venv fire_env
fire_env\Scripts\activate  # Windows
# source fire_env/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Additional geospatial dependencies (if needed)
conda install -c conda-forge gdal geopandas rasterio
```

### requirements.txt Content
```text
tensorflow>=2.12.0,<2.16.0
torch>=1.13.0
rasterio>=1.3.0
geopandas>=0.12.0
numpy>=1.21.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.0.0
opencv-python>=4.7.0
scikit-image>=0.19.0
scikit-learn>=1.2.0
netcdf4>=1.6.0
h5py>=3.7.0
pyyaml>=6.0
shapely>=2.0.0
pyproj>=3.4.0
fiona>=1.8.0
rtree>=1.0.0
tqdm>=4.64.0
```

## 🎯 Import Organization by Module

### data_preprocessing.py
```python
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import os
import logging
```

### fire_prediction_model.py
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
```

### fire_spread_simulation.py
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from scipy import ndimage
import rasterio
```

### utils.py
```python
import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
```

## 🚨 Common Installation Issues & Solutions

### GDAL Installation Issues
```bash
# Windows - Use conda for GDAL
conda install -c conda-forge gdal

# Alternative - Use OSGeo4W
# Download from https://trac.osgeo.org/osgeo4w/
```

### TensorFlow GPU Support
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Management
```python
# For large datasets, use these settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GDAL_CACHEMAX'] = '512'

# TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 📊 Hardware Requirements

### Minimum System Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **CPU**: 4 cores (8 cores recommended)
- **GPU**: Optional but recommended (4GB VRAM)

### Optimal Configuration
- **RAM**: 32GB+ for large-scale processing
- **Storage**: SSD with 100GB+ free space
- **CPU**: 16+ cores for parallel processing
- **GPU**: 8GB+ VRAM (RTX 3070 or better)

## 🔍 Dependency Purposes Summary

| Library | Primary Purpose | Fire Prediction Use |
|---------|----------------|-------------------|
| TensorFlow | Deep Learning | U-Net, LSTM models |
| Rasterio | Raster I/O | Reading satellite data |
| GeoPandas | Vector data | Road networks, boundaries |
| NumPy | Numerical computing | Array operations |
| OpenCV | Image processing | Fire spread visualization |
| Matplotlib | Plotting | Results visualization |
| Scikit-learn | ML utilities | Model evaluation |
| NetCDF4 | Weather data | Climate datasets |

This comprehensive guide ensures all team members understand the technical stack and can successfully set up and maintain the forest fire prediction system.

---

# 🚀 Quick Start Guide for Forest Fire Prediction System

## 📋 What I've Added:

### 1. Updated README2.md - Added complete "How to Run" section
### 2. New INSTALLATION_GUIDE.md - Step-by-step installation guide
### 3. New QUICK_RUN.py - Simplified execution script with error handling

## 🚀 For New Users - Simple Steps:

### Option 1: Quick Start (Recommended)
```bash
# 1. Clone/download the project
# 2. Open PowerShell in project folder
# 3. Run one command:
python QUICK_RUN.py
```

### Option 2: Manual Setup
```bash
# 1. Install Python 3.12
# 2. Create virtual environment
python -m venv fire_env
fire_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline
python src/main.py
```

### Option 3: With Sample Data
```bash
# Generate sample data first
python quick_start.py

# Then run full pipeline
python src/main.py --config config/default_config.yaml
```

## 📁 Complete File Structure:

```
TEAM ALFA/
├── README.md                    # Project overview
├── README2.md                   # Technical details & how to run
├── INSTALLATION_GUIDE.md        # Step-by-step setup guide
├── QUICK_RUN.py                 # One-click execution script
├── requirements.txt             # Dependencies list
├── quick_start.py              # Sample data generator
├── src/main.py                 # Main pipeline
└── config/default_config.yaml  # Configuration
```

## 🎯 Key Features of New Guides:

### INSTALLATION_GUIDE.md includes:
- ✅ Python version requirements
- ✅ Virtual environment setup
- ✅ Dependency installation (with troubleshooting)
- ✅ Hardware requirements
- ✅ Common error solutions

### QUICK_RUN.py features:
- ✅ Automatic environment checking
- ✅ Dependency installation if missing
- ✅ Sample data generation
- ✅ Error handling and user-friendly messages
- ✅ Progress indicators

### README2.md now includes:
- ✅ Complete "How to Run This Project" section
- ✅ Multiple execution options
- ✅ Expected outputs description
- ✅ Troubleshooting guide

## 💡 For Your Team:

**New users can now:**
1. Download the project
2. Run `python QUICK_RUN.py`
3. Get a working fire prediction system in minutes

**Advanced users can:**
1. Follow the detailed installation guide
2. Customize configurations
3. Use their own data

This makes your project much more accessible to new users while maintaining flexibility for advanced users!
