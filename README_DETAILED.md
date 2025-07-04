# Forest Fire Prediction and Spread Simulation - Technical Documentation

A comprehensive machine learning pipeline for forest fire prediction and simulation using deep learning and geospatial analysis.

## 🎯 Overview

This project implements an end-to-end forest fire prediction system that combines machine learning models with geospatial data processing to predict fire risk and simulate fire spread patterns. The system uses multiple environmental data layers to train deep learning models and provides real-time fire spread simulations.

## 🔧 Technical Stack & Dependencies

### Core Machine Learning Libraries

#### **TensorFlow** 
- **Purpose**: Deep learning framework for building and training neural networks
- **Why Used**: 
  - Provides U-Net convolutional neural networks for spatial fire prediction
  - LSTM networks for temporal fire pattern analysis
  - GPU acceleration for faster training
  - Keras high-level API for rapid model development
- **Usage**: Model building, training, and inference

#### **PyTorch**
- **Purpose**: Alternative deep learning framework 
- **Why Used**: 
  - Flexible dynamic computation graphs
  - Advanced model architectures
  - Research-oriented features
- **Usage**: Experimental models and custom neural network architectures

### Geospatial Data Processing

#### **Rasterio**
- **Purpose**: Reading and writing raster datasets (GeoTIFF files)
- **Why Used**: 
  - Handle satellite imagery and environmental data
  - Raster reprojection and resampling
  - Efficient memory management for large datasets
- **Usage**: Loading temperature, humidity, DEM, and other raster data

#### **GeoPandas**
- **Purpose**: Geospatial data manipulation and analysis
- **Why Used**: 
  - Process vector data (roads, boundaries, settlements)
  - Spatial operations and geometric calculations
  - Integration with pandas for data analysis
- **Usage**: Road networks, administrative boundaries processing

#### **Shapely**
- **Purpose**: Geometric operations on spatial objects
- **Why Used**: 
  - Create and manipulate geometric shapes
  - Spatial relationship analysis
  - Buffer operations for proximity calculations
- **Usage**: Distance calculations, geometric transformations

#### **Pyproj**
- **Purpose**: Cartographic projections and coordinate transformations
- **Why Used**: 
  - Convert between different coordinate systems
  - Ensure spatial data alignment
  - Accurate distance and area calculations
- **Usage**: CRS transformations, projection handling

### Scientific Computing

#### **NumPy**
- **Purpose**: Numerical computing with N-dimensional arrays
- **Why Used**: 
  - Fast array operations for raster data
  - Mathematical functions for data processing
  - Memory efficient data structures
- **Usage**: Array manipulation, mathematical operations

#### **SciPy**
- **Purpose**: Scientific computing and advanced mathematics
- **Why Used**: 
  - Image processing functions (ndimage)
  - Statistical functions
  - Optimization algorithms
- **Usage**: Image filtering, distance transforms, interpolation

#### **Pandas**
- **Purpose**: Data manipulation and analysis
- **Why Used**: 
  - Tabular data handling
  - Time series analysis
  - Data cleaning and preprocessing
- **Usage**: Metadata processing, statistics, data export

#### **Scikit-learn**
- **Purpose**: Machine learning utilities and algorithms
- **Why Used**: 
  - Data preprocessing (normalization, scaling)
  - Model evaluation metrics
  - Train/test splitting
- **Usage**: Data validation, performance metrics, preprocessing

### Visualization

#### **Matplotlib**
- **Purpose**: Plotting and visualization
- **Why Used**: 
  - Create maps and charts
  - Model performance visualization
  - Scientific plotting capabilities
- **Usage**: Training curves, prediction maps, statistical plots

#### **Seaborn**
- **Purpose**: Statistical data visualization
- **Why Used**: 
  - Beautiful statistical plots
  - Correlation matrices
  - Distribution analysis
- **Usage**: Data exploration, model analysis visualizations

### Computer Vision

#### **OpenCV (cv2)**
- **Purpose**: Computer vision and image processing
- **Why Used**: 
  - Image morphological operations
  - Contour detection for fire boundaries
  - Image filtering and enhancement
- **Usage**: Fire spread boundary detection, image processing

#### **Pillow (PIL)**
- **Purpose**: Image processing library
- **Why Used**: 
  - Image format conversions
  - Basic image operations
  - Thumbnail generation
- **Usage**: Image format handling, basic transformations

### Animation and Media

#### **ImageIO**
- **Purpose**: Reading and writing image and video files
- **Why Used**: 
  - Create animated GIFs and MP4 videos
  - Frame-by-frame animation creation
  - Multiple format support
- **Usage**: Fire spread animation generation

#### **ImageIO-FFMPEG**
- **Purpose**: Video encoding and decoding
- **Why Used**: 
  - High-quality MP4 video creation
  - Video compression and optimization
  - Professional video output
- **Usage**: Final animation rendering

### Configuration and Data Handling

#### **PyYAML**
- **Purpose**: YAML file parsing and generation
- **Why Used**: 
  - Human-readable configuration files
  - Easy parameter management
  - Hierarchical configuration structure
- **Usage**: Loading configuration parameters

#### **tqdm**
- **Purpose**: Progress bars for loops and processes
- **Why Used**: 
  - Visual feedback for long-running processes
  - Processing time estimation
  - User experience improvement
- **Usage**: Data processing progress, training progress

### Development and Notebooks

#### **Jupyter**
- **Purpose**: Interactive computing environment
- **Why Used**: 
  - Data exploration and analysis
  - Model development and testing
  - Documentation and tutorials
- **Usage**: Exploratory data analysis, model prototyping

#### **ipykernel**
- **Purpose**: Jupyter kernel for IPython
- **Why Used**: 
  - Enable Jupyter notebook functionality
  - Interactive Python execution
  - Integration with development environment
- **Usage**: Notebook execution environment

## 📊 Data Requirements

### Required Raw Data Files

| **Category** | **File** | **Format** | **Description** | **Units/Values** |
|--------------|----------|------------|-----------------|------------------|
| **Weather** | `temperature.tif` | GeoTIFF | Surface temperature | °C (Celsius) |
| **Weather** | `humidity.tif` | GeoTIFF | Relative humidity | % (0-100) |
| **Weather** | `wind_speed.tif` | GeoTIFF | Wind speed | m/s |
| **Weather** | `wind_direction.tif` | GeoTIFF | Wind direction | degrees (0-360) |
| **Weather** | `rainfall.tif` | GeoTIFF | Precipitation | mm |
| **Terrain** | `dem.tif` | GeoTIFF | Digital Elevation Model | meters above sea level |
| **Land Cover** | `landcover.tif` | GeoTIFF | Land use classification | Integer codes (1-8) |
| **Fire History** | `fire_2020.tif` | GeoTIFF | Fire occurrence 2020 | Binary (0/1) |
| **Fire History** | `fire_2021.tif` | GeoTIFF | Fire occurrence 2021 | Binary (0/1) |
| **Fire History** | `fire_2022.tif` | GeoTIFF | Fire occurrence 2022 | Binary (0/1) |
| **Infrastructure** | `roads.shp` | Shapefile | Road network (optional) | Vector geometry |
| **Infrastructure** | `settlements.shp` | Shapefile | Settlements (optional) | Vector geometry |

### Land Cover Classification
```
1 = Urban/Built-up areas
2 = Agricultural land
3 = Forest/Woody vegetation
4 = Grassland
5 = Shrubland
6 = Water bodies
7 = Barren land
8 = Wetland
```

## 🔄 Data Processing Pipeline

### 1. Data Preprocessing
- **Raster Alignment**: All data layers aligned to common grid
- **Reprojection**: Convert to consistent coordinate system (EPSG:4326)
- **Resampling**: Standardize to target resolution (0.0003 degrees ≈ 30m)
- **Normalization**: Scale values to [0,1] range for model training

### 2. Feature Engineering
- **Slope/Aspect**: Calculated from DEM using gradient analysis
- **Fuel Load**: Derived from land cover classification
- **Distance to Roads**: Euclidean distance calculation
- **Weather Normalization**: Statistical normalization of meteorological data

### 3. Model Training Data
- **Feature Stack**: 8-band raster with all environmental variables
- **Labels**: Binary fire occurrence from historical data
- **Spatial Patches**: 256x256 pixel patches for U-Net training
- **Temporal Sequences**: 7-day sequences for LSTM training

## 🤖 Machine Learning Models

### U-Net Architecture
```python
# Convolutional Neural Network for spatial prediction
Input: (256, 256, 8) - 8 environmental bands
Encoder: Downsampling path with skip connections
Bottleneck: Feature compression and representation
Decoder: Upsampling path with concatenated features
Output: (256, 256, 1) - Fire probability map
```

### LSTM Architecture
```python
# Recurrent Neural Network for temporal patterns
Input: (sequence_length=7, n_features=8)
LSTM Layers: 2 layers with 128 hidden units each
Dense Layer: Fully connected output layer
Output: Binary fire probability (0-1)
```

## 🔥 Fire Spread Simulation

### Cellular Automata Model
- **Grid-based**: Each cell represents 30m x 30m area
- **State Transitions**: Unburned → Burning → Burned
- **Factors**:
  - Wind speed and direction (acceleration)
  - Fuel load (combustibility)
  - Terrain slope (uphill spread)
  - Moisture content (resistance)

### Mathematical Model
```
Fire Spread Rate = Base Rate × Wind Effect × Fuel Effect × Slope Effect × (1 - Moisture Effect)
```

## 📈 Output Products

### 1. Fire Risk Maps
- **Format**: GeoTIFF raster files
- **Resolution**: 30m spatial resolution
- **Values**: Probability [0-1] of fire occurrence
- **Projection**: EPSG:4326 (WGS84)

### 2. Fire Spread Animations
- **Format**: MP4 video files
- **Frame Rate**: 2 FPS (configurable)
- **Duration**: Simulation time period
- **Content**: Dynamic fire spread visualization

### 3. Statistical Reports
- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **Simulation Statistics**: Affected area, spread rate, perimeter
- **Time Series**: Hourly progression data

## ⚙️ Configuration Parameters

### Model Configuration
```yaml
# Neural Network Parameters
batch_size: 8                    # Training batch size
epochs: 10                       # Training epochs
learning_rate: 0.001             # Adam optimizer learning rate
hidden_size: 128                 # LSTM hidden units
n_layers: 2                      # LSTM layers
sequence_length: 7               # Temporal sequence length

# Spatial Parameters
input_shape: [256, 256, 8]       # U-Net input dimensions
patch_size: [256, 256]           # Training patch size
target_resolution: 0.0003        # Spatial resolution (degrees)
```

### Simulation Configuration
```yaml
# Fire Spread Parameters
simulation_hours: 6.0            # Simulation duration
time_step: 300                   # Time step (seconds)
base_spread_rate: 0.1            # Base fire spread rate
wind_effect: 0.5                 # Wind influence factor
fuel_effect: 0.8                 # Fuel load influence
slope_effect: 0.3                # Terrain slope influence
moisture_effect: 0.6             # Moisture resistance factor
```

## 🚀 Installation and Setup

### System Requirements
- **Python**: 3.12 or earlier (TensorFlow compatibility)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB for dependencies, additional space for data
- **GPU**: Optional but recommended for faster training

### Installation Steps
```bash
# 1. Clone repository
git clone <repository-url>
cd TEAM_ALFA

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

## 🏃‍♂️ Usage Examples

### Basic Pipeline Execution
```bash
# Run complete pipeline
python src/main.py

# Run with custom configuration
python src/main.py --config config/my_config.yaml

# Run specific components
python src/main.py --skip-preprocessing
python src/main.py --skip-prediction
python src/main.py --skip-simulation
```

### Quick Start with Sample Data
```bash
# Generate sample data and run demo
python quick_start.py

# Create sample data only
python quick_start.py --create-data-only
```

### Jupyter Notebook Analysis
```bash
# Start Jupyter server
jupyter lab

# Open analysis notebooks
notebooks/data_analysis.ipynb
notebooks/model_training.ipynb
```

## 🔬 Scientific Background

### Fire Behavior Fundamentals
- **Fire Triangle**: Heat, Fuel, Oxygen requirements
- **Spread Mechanisms**: Radiation, convection, conduction
- **Environmental Factors**: Weather, topography, fuel characteristics

### Machine Learning Approach
- **Supervised Learning**: Historical fire data as ground truth
- **Feature Engineering**: Environmental variables as predictors
- **Spatial Context**: Convolutional networks for spatial patterns
- **Temporal Context**: Recurrent networks for time series

## 📚 References and Data Sources

### Recommended Data Sources
- **Weather**: ERA5 Reanalysis, MODIS LST, local weather stations
- **Terrain**: SRTM DEM, ASTER GDEM, national elevation datasets
- **Land Cover**: MODIS Land Cover, ESA WorldCover, national datasets
- **Fire Data**: MODIS/VIIRS Active Fire, local fire department records

### Scientific References
- Rothermel, R.C. (1972). A mathematical model for predicting fire spread
- Sullivan, A.L. (2009). Wildland surface fire spread modelling
- Jain, P. et al. (2020). A review of machine learning applications in wildfire science

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

### Code Standards
- **Style**: Follow PEP 8 Python style guide
- **Documentation**: Include docstrings for all functions
- **Testing**: Add unit tests for new features
- **Comments**: Explain complex algorithms and data flows

## 📞 Support and Contact

For technical support, bug reports, or feature requests:
- Create GitHub issues for bugs and features
- Check documentation for common questions
- Contact maintainers for collaboration opportunities

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

*This documentation provides comprehensive technical details for researchers, developers, and users working with the forest fire prediction and simulation system.*
