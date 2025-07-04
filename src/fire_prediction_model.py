"""
Fire prediction models using U-Net and LSTM architectures.
Implements deep learning models for binary fire risk classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import os
import logging
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class FireDataset(Dataset):
    """PyTorch Dataset for fire prediction data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature array
            labels: Label array
            transform: Optional data transformation
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'labels': self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class UNetFirePredictor:
    """U-Net model for spatial fire prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize U-Net model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Model parameters
        self.input_shape = config.get('input_shape', (256, 256, 10))
        self.n_classes = config.get('n_classes', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 16)
        self.epochs = config.get('epochs', 100)
        
    def build_model(self) -> keras.Model:
        """Build U-Net architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder (Contracting Path)
        c1 = self._conv_block(inputs, 64)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self._conv_block(p1, 128)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self._conv_block(p2, 256)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self._conv_block(p3, 512)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = self._conv_block(p4, 1024)
        
        # Decoder (Expanding Path)
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = self._conv_block(u6, 512)
        
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = self._conv_block(u7, 256)
        
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = self._conv_block(u8, 128)
        
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = self._conv_block(u9, 64)
        
        # Output layer
        if self.n_classes == 2:
            outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        else:
            outputs = layers.Conv2D(self.n_classes, (1, 1), activation='softmax')(c9)
            
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        
        # Compile model
        if self.n_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
        
    def _conv_block(self, inputs, filters: int):
        """Convolutional block with batch normalization and dropout."""
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        return x
        
    def prepare_data_for_unet(self, features: np.ndarray, labels: np.ndarray,
                             patch_size: Tuple[int, int] = (256, 256)) -> Tuple:
        """
        Prepare data for U-Net training (convert to image patches).
        
        Args:
            features: Feature raster data
            labels: Label raster data
            patch_size: Size of image patches
            
        Returns:
            Tuple of (X_patches, y_patches)
        """
        logger.info("Preparing data patches for U-Net...")
        
        # Reshape features to image format if needed
        if len(features.shape) == 2:
            # Single band to multi-band
            height, width = features.shape
            n_bands = 1
            features = features.reshape(height, width, n_bands)
        else:
            height, width, n_bands = features.shape
            
        # Extract patches
        patches_x = []
        patches_y = []
        
        patch_h, patch_w = patch_size
        
        for i in range(0, height - patch_h + 1, patch_h // 2):
            for j in range(0, width - patch_w + 1, patch_w // 2):
                patch_x = features[i:i+patch_h, j:j+patch_w, :]
                patch_y = labels[i:i+patch_h, j:j+patch_w]
                
                # Skip patches with too many nodata values
                if np.sum(np.isnan(patch_x)) < 0.1 * patch_x.size:
                    patches_x.append(patch_x)
                    patches_y.append(patch_y)
                    
        X_patches = np.array(patches_x)
        y_patches = np.array(patches_y)
        
        # Expand dimensions for binary classification
        if self.n_classes == 2:
            y_patches = np.expand_dims(y_patches, axis=-1)
            
        logger.info(f"Created {len(X_patches)} patches of size {patch_size}")
        
        return X_patches, y_patches
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str) -> keras.callbacks.History:
        """
        Train U-Net model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_save_path: Path to save trained model
            
        Returns:
            Training history
        """
        logger.info("Training U-Net model...")
        
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        callback_list = [
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("U-Net training completed")
        return self.history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        predictions = self.model.predict(X, batch_size=self.batch_size)
        return predictions
        
    def load_model(self, model_path: str):
        """Load trained model."""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")


class LSTMFirePredictor:
    """LSTM model for temporal fire prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize LSTM model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Model parameters
        self.sequence_length = config.get('sequence_length', 7)  # 7 days
        self.n_features = config.get('n_features', 10)
        self.hidden_size = config.get('hidden_size', 128)
        self.n_layers = config.get('n_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        
    def build_model(self) -> keras.Model:
        """Build LSTM architecture."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        x = layers.LSTM(self.hidden_size, return_sequences=True, dropout=0.2)(inputs)
        x = layers.LSTM(self.hidden_size // 2, return_sequences=False, dropout=0.2)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
        
    def prepare_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple:
        """
        Prepare sequential data for LSTM training.
        
        Args:
            features: Time series features (samples, time_steps, features)
            labels: Time series labels
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        logger.info("Preparing sequences for LSTM...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(features)):
            X_sequences.append(features[i-self.sequence_length:i])
            y_sequences.append(labels[i])
            
        return np.array(X_sequences), np.array(y_sequences)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str) -> keras.callbacks.History:
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        if self.model is None:
            self.build_model()
            
        # Define callbacks
        callback_list = [
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("LSTM training completed")
        return self.history


class ModelEvaluator:
    """Model evaluation and metrics calculation."""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Basic classification metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'confusion_matrix': cm.tolist()
        }
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        self.metrics = metrics
        
        # Save metrics
        if save_path:
            import json
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        # Print summary
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
        return metrics
        
    def plot_training_history(self, history: keras.callbacks.History, 
                            save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Fire', 'Fire'],
                   yticklabels=['No Fire', 'Fire'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def create_fire_prediction_pipeline(config: Dict) -> Dict:
    """
    Create complete fire prediction pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing trained models and results
    """
    logger.info("Creating fire prediction pipeline...")
    
    # Initialize components
    if config['model_type'] == 'unet':
        model = UNetFirePredictor(config)
    elif config['model_type'] == 'lstm':
        model = LSTMFirePredictor(config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
        
    evaluator = ModelEvaluator()
    
    # Load and prepare data
    from data_preprocessing import FireDataPreprocessor
    preprocessor = FireDataPreprocessor(config)
    
    # Get processed data paths
    processed_paths = preprocessor.run_preprocessing_pipeline(config)
    
    # Prepare training data
    features, labels = preprocessor.prepare_training_data(
        processed_paths['feature_stack'],
        processed_paths['fire_history']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Prepare data for specific model type
    if config['model_type'] == 'unet':
        # Convert to image patches for U-Net
        X_train, y_train = model.prepare_data_for_unet(X_train, y_train)
        X_val, y_val = model.prepare_data_for_unet(X_val, y_val)
        X_test, y_test = model.prepare_data_for_unet(X_test, y_test)
        
    elif config['model_type'] == 'lstm':
        # Prepare sequences for LSTM
        X_train, y_train = model.prepare_sequences(X_train, y_train)
        X_val, y_val = model.prepare_sequences(X_val, y_val)
        X_test, y_test = model.prepare_sequences(X_test, y_test)
        
    # Train model
    model_save_path = os.path.join(config['paths']['models'], f"{config['model_type']}_model.h5")
    history = model.train(X_train, y_train, X_val, y_val, model_save_path)
    
    # Evaluate model
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics_path = os.path.join(config['paths']['outputs'], 'metrics', 'evaluation_metrics.json')
    metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba, metrics_path)
    
    # Plot results
    plots_dir = os.path.join(config['paths']['outputs'], 'visualizations')
    os.makedirs(plots_dir, exist_ok=True)
    
    evaluator.plot_training_history(history, 
                                  os.path.join(plots_dir, 'training_history.png'))
    evaluator.plot_confusion_matrix(y_test, y_pred,
                                  os.path.join(plots_dir, 'confusion_matrix.png'))
    
    logger.info("Fire prediction pipeline completed successfully")
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'evaluator': evaluator
    }
