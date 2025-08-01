# HazeHeal: Advanced Image Dehazing Model

![project4](https://github.com/user-attachments/assets/cac61cb5-91a2-4228-9e09-cce4b7eb0299)


HazeHeal is a state-of-the-art GAN-based image dehazing model implemented in TensorFlow 2.x. It effectively removes haze from images while preserving the original image details and colors.

## Architecture Overview

The model consists of two main networks:

### Generator

The generator network features:
- **Swish activation** in the encoder for smooth gradient flow  
- **MBConv (Mobile Inverted Bottleneck) blocks** for lightweight and efficient feature extraction  
- **U-Net style encoder-decoder architecture** with skip connections  
- **Conv2DTranspose layers** for learnable upsampling and high-quality reconstruction  
- **Multi-scale fusion** of standard convolutional and bottleneck block outputs  

### Discriminator

The discriminator network includes:
- **LeakyReLU activation** (α = 0.2) for consistent gradient propagation  
- **Spectral normalization** for training stability across layers  
- **No batch normalization** to maintain adversarial training dynamics  
- **PatchGAN architecture** that outputs spatial prediction maps  
- **Conditional design** using concatenated input-target image pairs  

### Features

- **Advanced Normalization**: Spectral normalization stabilizes discriminator updates  
- **Mixed Activations**: Swish in the generator encoder; LeakyReLU in the discriminator  
- **Efficient Architectures**: Uses MBConv blocks for efficient representation learning  
- **Structured Design**: U-Net generator with skip connections for better information flow  
- **Conditional PatchGAN**: Discriminator evaluates joint input-output image realism  
- **Multi-Scale Processing**: Fuses mobile bottlenecks with standard convolutions  
- **Learnable Upsampling**: Transposed convolutions for sharper image reconstruction  

### Loss Functions

- **Charbonnier Loss**: A smooth L1 variant that better preserves image edges  
- **Haze-Aware Loss**: Adaptive loss weighting based on haze masks to focus on difficult regions  
- **Perceptual Loss**: Encourages high-level texture realism by comparing deep features  
- **Adversarial Loss**: Drives the generator to produce photorealistic outputs  

### Optimization

- **Adaptive Loss Weighting**: Weighted combination of reconstruction (λ = 150) and perceptual loss (λ = 0.005) for balanced learning  

## Requirements

```
tensorflow >= 2.0
numpy
matplotlib
pillow
glob
logging
```

## Project Structure

```
HazeHeal/
├── checkpoints/         
├── samples/             
├── generator_model.png  
├── discriminator_model.png 
├── HazeHeal.ipynb             
└── README.md           
```

## Usage

### Data Preparation

Place your training data in the following structure:
```
dataset/
├── hazy/              # Hazy images
└── GT/                # Ground truth clear images
```

### Training

```python
# Configure training parameters
HAZY_DIR = '/path/to/hazy/images'
GT_DIR = '/path/to/ground/truth/images'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 200


```

### Model Components

#### DehazeGAN

Main model class implementing the GAN architecture:
- Custom generator with U-Net structure and MBConv blocks  
- PatchGAN discriminator with spectral normalization  
- Multi-component loss functions: Charbonnier, Perceptual, Adversarial  

#### DehazeDataProcessor

Handles data loading and preprocessing:
- Image resizing and normalization  
- Dataset creation and batching  
- Data augmentation for generalization  

#### DehazeTrainer

Manages the training process:
- Checkpoint handling and restoration  
- Progress visualization with sample outputs  
- Generates dehazed image samples during training  
  

### Training Process

The training pipeline includes:
- Haze-aware loss computation using adaptive region weighting  
- Multi-component loss optimization balancing reconstruction and perceptual quality  
- Spectral normalization in the discriminator for stability  
- Regular checkpoint saving for recovery and reproducibility  
- Sample image generation every 10 epochs  
- Real-time progress logging with detailed loss metrics  
- Visualization of input, predicted, and target images during training  


### Loss Function Details

The generator optimizes a sophisticated loss combination:
- **Adversarial Loss**: Standard GAN loss encouraging realism  
- **Charbonnier Loss**: Robust reconstruction loss (λ = 150)  
- **Haze-Aware Charbonnier**: Region-weighted loss (70% global + 30% masked)  
- **Perceptual Loss**: VGG-based feature matching


## Model Performance

The model achieves optimal dehazing results after approximately 2500 epochs, with:
- Generator Loss: 8.5700
- Discriminator Loss: 1.084




