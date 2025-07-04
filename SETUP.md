# Forest Fire Prediction and Spread Simulation - Setup Instructions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM recommended
- GPU optional but recommended for faster training

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv fire_env
source fire_env/bin/activate  # On Windows: fire_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Quick Demo with Sample Data

The fastest way to test the pipeline is using the provided sample data generator:

```bash
# Generate sample data and run complete pipeline
python quick_start.py

# Or just create sample data without running pipeline
python quick_start.py --create-data-only
```

This will create realistic sample environmental data and run the entire pipeline.

### 3. Using Your Own Data

#### Required Data Files:
Place your raster files in `data/raw/` directory:

**Weather Data:**
- `temperature.tif` - Temperature in Celsius
- `humidity.tif` - Relative humidity (0-100%)
- `wind_speed.tif` - Wind speed in m/s
- `wind_direction.tif` - Wind direction in degrees (0-360)
- `rainfall.tif` - Rainfall in mm

**Terrain Data:**
- `dem.tif` - Digital Elevation Model in meters

**Land Cover Data:**
- `landcover.tif` - Land use/land cover classification

**Historical Fire Data:**
- `fire_2020.tif`, `fire_2021.tif`, `fire_2022.tif` - Binary fire occurrence maps

#### Update Configuration:
Edit `config/default_config.yaml` to point to your data files:

```yaml
data_paths:
  temperature: "data/raw/your_temperature.tif"
  humidity: "data/raw/your_humidity.tif"
  # ... update other paths
```

### 4. Run the Pipeline

```bash
# Run complete pipeline
python src/main.py --config config/default_config.yaml

# Run specific steps only
python src/main.py --skip-preprocessing  # Skip data preprocessing
python src/main.py --skip-prediction    # Skip model training
python src/main.py --skip-simulation    # Skip fire spread simulation
```

## 📁 Output Structure

After running the pipeline, you'll find results in `data/outputs/`:

```
data/outputs/
├── predictions/           # Fire risk maps and ignition points
├── simulations/          # Fire spread simulation GeoTIFFs
├── animations/           # MP4 animations of fire spread
├── metrics/              # Model performance metrics and statistics
└── visualizations/       # Plots and analysis charts
```

## 🔧 Configuration Options

### Model Selection
Choose between U-Net (spatial) or LSTM (temporal) models:

```yaml
model_type: "unet"  # or "lstm"
```

### Simulation Parameters
Adjust fire spread simulation settings:

```yaml
simulation_hours: 12.0        # Total simulation time
save_interval_hours: 1.0      # Save interval
cell_size: 30                 # Cell resolution in meters
base_spread_rate: 0.1         # Base fire spread rate (m/s)
```

### Training Parameters
Modify model training settings:

```yaml
batch_size: 16
epochs: 100
learning_rate: 0.001
```

## 📊 Using Jupyter Notebooks

Explore the data and models interactively:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks
# - notebooks/data_analysis.ipynb     # Data exploration and analysis
# - notebooks/model_training.ipynb   # Model training and evaluation
```

## 🔍 Data Requirements

### Raster Data Format
- **Format**: GeoTIFF (.tif)
- **Resolution**: Preferably 30m (will be resampled if different)
- **CRS**: Any standard CRS (will be reprojected to target CRS)
- **Extent**: All rasters should cover the same study area

### Data Quality
- Minimize missing data (NoData values)
- Ensure temporal consistency for historical fire data
- Validate coordinate systems and extents

### Land Cover Encoding
Default fuel load mapping (modify in config if needed):
- 1: Urban/Built-up (0.1)
- 2: Agricultural (0.3)
- 3: Forest/Woody (0.8)
- 4: Grassland (0.6)
- 5: Shrubland (0.9)
- 6: Water (0.0)
- 7: Barren (0.2)

## ⚡ Performance Tips

### For Large Datasets:
1. **Preprocessing**: Run preprocessing once, then skip for subsequent runs
2. **Patch Size**: Reduce U-Net patch size if memory issues occur
3. **Batch Size**: Reduce batch size for GPU memory constraints
4. **Simulation**: Reduce simulation time or increase save intervals

### Hardware Recommendations:
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for large study areas
- **GPU**: NVIDIA GPU with 8GB+ VRAM for faster training
- **Storage**: SSD recommended for faster I/O

## 🐛 Troubleshooting

### Common Issues:

**1. Memory Errors:**
```bash
# Reduce batch size in config
batch_size: 4

# Or reduce input resolution
target_resolution: 100  # Instead of 30
```

**2. Missing Dependencies:**
```bash
# Install GDAL separately if issues
conda install gdal

# Or use conda environment
conda env create -f environment.yml  # If provided
```

**3. CRS/Projection Issues:**
- Ensure all raster files have valid CRS information
- Use `gdalinfo` to check raster properties
- Reproject data to common CRS if needed

**4. No GPU Detected:**
```bash
# Install TensorFlow with GPU support
pip install tensorflow-gpu

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📚 Understanding Outputs

### Fire Risk Maps
- **Values**: 0-1 probability of fire occurrence
- **High Risk**: Values > 0.8 typically used as ignition points
- **Format**: GeoTIFF with same CRS as input data

### Fire Spread Simulations
- **Time Series**: Multiple GeoTIFF files for each time step
- **Values**: 0=Unburned, 1=Burning, 2=Burned, 3=No Fuel
- **Animation**: MP4 video showing fire progression

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to fire events
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under receiver operating curve

## 🎯 Next Steps

1. **Model Improvement**: Tune hyperparameters based on your data
2. **Feature Engineering**: Add additional environmental variables
3. **Validation**: Test on independent datasets
4. **Integration**: Integrate with operational fire management systems
5. **Real-time**: Adapt for real-time weather data feeds

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files (`fire_prediction.log`)
3. Examine intermediate outputs in `data/processed/`
4. Test with sample data first to isolate issues

## 📖 References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- LSTM: [Hochreiter & Schmidhuber, 1997](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
- Cellular Automata for Fire Modeling: [Alexandridis et al., 2008](https://www.sciencedirect.com/science/article/pii/S0307904X07001473)
