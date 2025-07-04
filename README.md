# Forest Fire Prediction and Spread Simulation

A machine learning pipeline for predicting forest fire risk and simulating fire spread using raster datasets and deep learning models.

## 🌲 Project Overview

This project implements a comprehensive forest fire prediction and simulation system that:
- Predicts binary fire risk maps using U-Net/LSTM models
- Simulates fire spread using Cellular Automata
- Processes multiple geospatial data layers at 30m resolution
- Outputs GeoTIFF prediction maps and animated fire spread simulations

## 📊 Input Data Layers

- **Weather Parameters**: Temperature, humidity, wind speed/direction, rainfall
- **Terrain Parameters**: Slope, aspect (derived from DEM)
- **Land Cover Data**: LULC as fuel proxy
- **Human Activity**: Road proximity, settlements
- **Historical Fire Data**: VIIRS/MODIS fire occurrences

## 🎯 Key Features

1. **Fire Risk Prediction**: Binary classification (fire/no fire) for next day
2. **Fire Spread Simulation**: Dynamic spread over 1, 2, 3, 6, and 12 hours
3. **Multi-resolution Processing**: 30m raster alignment and resampling
4. **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score metrics
5. **Visualization**: Interactive maps and animated fire spread

## 🏗️ Project Structure

```
forest_fire_prediction/
├── src/
│   ├── data_preprocessing.py    # Raster processing and alignment
│   ├── fire_prediction_model.py # U-Net/LSTM model definitions
│   ├── fire_spread_simulation.py # Cellular Automata implementation
│   ├── utils.py                 # Utility functions
│   └── main.py                  # Main pipeline execution
├── data/
│   ├── raw/                     # Raw input raster files
│   ├── processed/               # Preprocessed and aligned rasters
│   └── outputs/                 # Prediction and simulation results
├── models/                      # Trained model weights
├── config/                      # Configuration files
├── notebooks/                   # Jupyter notebooks for analysis
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your raster files in `data/raw/` directory

3. **Run Pipeline**:
   ```bash
   python src/main.py --config config/default_config.yaml
   ```

## 📈 Outputs

- **Fire Prediction**: GeoTIFF raster maps showing fire risk probability
- **Fire Spread Simulation**: Time-series GeoTIFFs and MP4 animations
- **Evaluation Metrics**: Model performance statistics
- **Visualizations**: Interactive maps and plots

## 🔧 Configuration

Modify `config/default_config.yaml` to adjust:
- Input data paths
- Model parameters
- Simulation settings
- Output formats

## 📚 Models

### Fire Prediction Models
- **U-Net**: Convolutional neural network for spatial prediction
- **LSTM**: Long Short-Term Memory for temporal patterns

### Fire Spread Simulation
- **Cellular Automata**: Dynamic fire spread modeling based on:
  - Wind conditions
  - Fuel load
  - Terrain slope
  - Moisture content

## 📝 License

This project is licensed belongs to Pratul Kumar

## 👥 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
