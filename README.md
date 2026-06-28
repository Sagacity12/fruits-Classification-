# fruits-Classification-
# Fruit Classification using Transfer Learning

A deep learning project that classifies fruits using transfer learning with VGG16 convolutional neural network, trained on the Fruits-360 dataset.

## Overview

This project implements a fruit image classification system using:
- **Transfer Learning** with VGG16 pre-trained on ImageNet
- **Fine-tuning** of the last 5 layers for improved accuracy
- **Data Augmentation** to enhance model generalization
- **Mixed Precision Training** for optimized performance

## Dataset

The project uses the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) which contains:
- 131 fruit and vegetable categories
- High-quality images with consistent backgrounds
- Separate training, validation, and test sets

### Dataset Structure
```
Dataset/
└── fruits-360_original-size/
    └── fruits-360-original-size/
        ├── Training/
        ├── Validation/
        └── Test/
```

## Features

- **Transfer Learning**: Leverages VGG16 pre-trained weights
- **Two-Phase Training**:
  - Initial training with frozen base layers
  - Fine-tuning with unfrozen last 5 layers
- **Data Augmentation**: Rotation, shifting, shearing, zooming, and flipping
- **Advanced Callbacks**:
  - Learning rate reduction on plateau
  - Early stopping to prevent overfitting
- **Visualization**: Training curves and sample predictions
- **Mixed Precision**: Optimized for GPU training

## Model Architecture

```
VGG16 Base (frozen initially)
    ↓
Global Average Pooling 2D
    ↓
Dense (256, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense (num_classes, Softmax)
```

## Requirements

```
tensorflow>=2.10.0
numpy
pandas
matplotlib
sympy
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fruit-Classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib sympy
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) and extract it to the `Dataset/` directory

## Usage

Run the main training script:
```bash
python main.py
```

The script will:
1. Load and preprocess the dataset
2. Train the model in two phases (initial + fine-tuning)
3. Evaluate on the test set
4. Display accuracy/loss curves
5. Show predictions on sample images

## Training Configuration

- **Image Size**: 224×224×3
- **Batch Size**: 32 (training), 16 (validation/test)
- **Epochs**: 5 (initial) + 5 (fine-tuning)
- **Optimizer**: Adam (1e-5 learning rate for fine-tuning)
- **Loss Function**: Categorical Crossentropy
- **Precision**: Mixed Float16

## Results

The model outputs:
- Test accuracy metrics
- Training/validation accuracy curves
- Training/validation loss curves
- Visual predictions with actual vs predicted labels

## Project Files

- `main.py` - Main training and evaluation script
- `file_extractions.py` - Dataset download utilities (commented out)
- `.gitignore` - Git ignore configuration
- `Dataset/` - Dataset directory
- `.venv/` - Virtual environment

## Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size** in data generators
2. **Use MobileNetV2** instead of VGG16:
   ```python
   from tensorflow.keras.applications import MobileNetV2
   base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
   ```
3. **Limit TensorFlow memory**:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

## License

This project is available for educational and research purposes.

## Acknowledgments

- Fruits-360 dataset by Horea Muresan and Mihai Oltean
- VGG16 architecture from the Visual Geometry Group, Oxford
- TensorFlow/Keras framework
