# Neural Style Transfer

## Project Overview
This project implements **Neural Style Transfer** using a pre-trained **VGG-19** deep learning model. It combines the content of one image with the artistic style of another image to generate a new stylized image.

## Input Images
- **Style Image:** `vangogh_starry_night.jpg`
  - Provides the artistic style and texture.
- **Content Image:** `Tuebingen_Neckarfront.jpg`
  - Provides the structure and content of the final image.
- Both images are resized to **512 × 512**.

## Deep Learning Model
- **VGG-19**
- Used as a pre-trained feature extractor.
- The model extracts content and style features from different convolutional layers.
- VGG weights are loaded from `vgg_conv.pth`.

## Architecture

```text
Style Image
     │
     ▼
   VGG-19
     │
     ▼
Style Feature Layers
(r11, r21, r31, r41, r51)
     │
     ▼
 Gram Matrix
     │
     ▼
 Style Loss
     │
     ├──────────────┐
     │              │
     │              ▼
     │        Total Loss
     │              ▲
     │              │
     ▼              │
Content Image       │
     │              │
     ▼              │
   VGG-19           │
     │              │
     ▼              │
Content Layer       │
   (r42)            │
     │              │
     ▼              │
 Content Loss ──────┘
     │
     ▼
L-BFGS Optimization
     │
     ▼
Generated Stylized Image
VGG Layers Used

## Style Layers:

r11
r21
r31
r41
r51

##Content Layer:

r42

The VGG architecture contains convolutional layers with 3 × 3 kernels, ReLU activation and 2 × 2 max pooling.

## Working

Load the style and content images.
Resize the images to 512 × 512.
Convert the images into PyTorch tensors and normalize them.
Pass the images through the pre-trained VGG-19 network.
Extract style features from the selected style layers.
Calculate Gram matrices from the style features.
Extract content features from the r42 layer.
Initialize the generated image using the content image.
Calculate the style and content losses.
Combine the losses using their respective weights.
Optimize the generated image using the L-BFGS optimizer.
Continue optimization for up to 500 iterations.
Post-process the optimized tensor to obtain the final stylized image.

## Technologies Used

Python
PyTorch
TorchVision
PIL / Pillow
NumPy
Matplotlib
Jupyter Notebook
Loss Function

The project uses Style Loss and Content Loss.

## Style Loss

Style loss is calculated using the Gram Matrix and Mean Squared Error (MSE).

Style Loss = MSE(Generated Gram Matrix, Style Gram Matrix)

The Gram Matrix represents correlations between feature maps and captures the artistic style and texture.

## Content Loss

Content loss compares the content features of the generated image with the content image.

Content Loss = MSE(Generated Features, Content Features)
Total Loss
Total Loss = Weighted Style Loss + Weighted Content Loss

The total loss is minimized during optimization so that the generated image preserves the content structure while adopting the style of the style image.
