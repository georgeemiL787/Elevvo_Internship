# Music Genre Classification System

This project implements a comprehensive music genre classification system using both **tabular** (audio features) and **image-based** (spectrograms) approaches. The system can classify music into 10 different genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

##  Project Overview

The system implements three main approaches:

1. **Tabular Approach**: Uses extracted audio features (MFCCs, spectral features, etc.) with traditional ML algorithms
2. **Image-based Approach with Transfer Learning**: Uses pre-trained VGG16 on spectrogram images
3. **Custom CNN Approach**: Uses a custom convolutional neural network on spectrogram images

## Project Structure

```
Task 6/
├── music_genre_classification.py      # Main classification system
├── step_by_step_implementation.py     # Step-by-step testing script
├── examine_features.py                 # Utility to examine feature files
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── features_3_sec.csv                 # 3-second audio features
├── features_30_sec.csv                # 30-second audio features
├── genres_original/                    # Audio files (10 genres)
└── images_original/                    # Spectrogram images (10 genres)
```

Step-by-Step Implementation

### Step 1: Tabular Approach
- Loads pre-extracted audio features from CSV files
- Preprocesses and scales the features
- Trains Random Forest and SVM classifiers
- Evaluates performance using cross-validation

### Step 2: Image-based Approach with Transfer Learning
- Loads spectrogram images from the `images_original/` directory
- Preprocesses images (resize, normalize)
- Uses pre-trained VGG16 with transfer learning
- Implements data augmentation for better generalization

### Step 3: Custom CNN Approach
- Creates a custom convolutional neural network
- Trains from scratch on spectrogram images
- Implements batch normalization and dropout for regularization

### Step 4: Approach Comparison
- Compares all three approaches
- Identifies the best performing method
- Provides detailed performance metrics

##  Audio Features Used

The tabular approach extracts and uses:

- **MFCCs (Mel-frequency cepstral coefficients)**: Capture timbral characteristics
- **Spectral Centroid**: Measures brightness of sound
- **Chroma Features**: Capture harmonic content
- **Tempo**: Beat tracking information
- **Zero Crossing Rate**: Measure of noisiness

## Image Processing

The image-based approaches:

- Load spectrogram images (PNG format)
- Resize to 224x224 pixels (compatible with VGG16)
- Normalize pixel values to [0, 1] range
- Apply data augmentation (rotation, shifts, flips)

##  Model Architectures

### Tabular Models
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel

### Image Models
- **VGG16 Transfer Learning**: Pre-trained on ImageNet
- **Custom CNN**: 4 convolutional layers + dense layers

## Performance Metrics

The system evaluates models using:

- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed error analysis
- **Cross-validation**: Robust performance estimation

##  Customization Options

### Modify Model Parameters
```python
# In music_genre_classification.py
classifier = MusicGenreClassifier()
classifier.train_image_model(X_train, y_train, X_test, y_test, 
                           use_transfer_learning=True)  # or False
```

### Adjust Training Parameters
```python
# Modify epochs, batch size, etc.
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,  # Change number of epochs
    validation_data=(X_test, y_test),
    callbacks=callbacks
)
```

### Feature Engineering
```python
# Add custom features in extract_audio_features method
def extract_audio_features(self, audio_file, duration=30):
    # ... existing features ...
    
    # Add your custom features here
    # Example: Harmonic features, rhythm features, etc.
    
    return features
```

##  Expected Results

Based on typical performance for this dataset:

- **Tabular Approach**: 60-80% accuracy
- **Transfer Learning**: 70-85% accuracy  
- **Custom CNN**: 65-80% accuracy

*Note: Actual performance may vary based on data quality and model tuning*

##  Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or image resolution
2. **CUDA Errors**: Ensure TensorFlow GPU compatibility
3. **File Not Found**: Check file paths and permissions
4. **Import Errors**: Verify all dependencies are installed

### Performance Tips

1. **Use GPU**: Enable TensorFlow GPU support for faster training
2. **Reduce Data**: Start with smaller subset for testing
3. **Adjust Parameters**: Modify learning rate, batch size, etc.
4. **Data Augmentation**: Increase augmentation for better generalization

##  Advanced Usage

### Ensemble Methods
```python
# Combine multiple models for better performance
ensemble_pred = (rf_pred + svm_pred + cnn_pred) / 3
```

### Hyperparameter Tuning
```python
# Use GridSearchCV or RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
```

### Custom Data Loading
```python
# Load your own audio files
audio_files = ['path/to/audio1.wav', 'path/to/audio2.wav']
features = [classifier.extract_audio_features(f) for f in audio_files]
```

##  References

- **GTZAN Dataset**: The dataset used for this project
- **Librosa**: Audio and music analysis library
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning library

##  Contributing

Feel free to contribute by:
- Improving model architectures
- Adding new feature extraction methods
- Optimizing performance
- Adding new evaluation metrics

## License

This project is part of the Elevvo Internship Task 6.

---

**Happy Music Classification! **
