
# Advanced Plant Pathology Classification Using Deep Learning

## Project Overview
This repository contains a deep learning project aimed at classifying plant diseases from images of leaves. The project utilizes DenseNet121 and EfficientNet models, incorporating modifications and ensemble techniques to optimize performance for plant pathology diagnostics.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training Procedures](#training-procedures)
- [Evaluation and Results](#evaluation-and-results)
- [Visualizations](#visualizations)
- [Usage Example](#usage-example)
- [Contributing](#contributing)
- [Citations](#citations)
- [License](#license)

## Installation
To run this project, install the required libraries by executing the following commands:

\```bash
pip install torch torchvision
pip install plotly matplotlib scikit-learn pandas
\```

## Dataset

This project utilizes a curated dataset comprising various categories of plant leaf images labeled as Healthy, Multiple Diseases, Rust, and Scab. The dataset aims to reflect real-world conditions and is designed to train robust models capable of handling diverse plant conditions

### Data Splitting

The dataset is rigorously split as follows to ensure the model’s ability to generalize well on unseen data:
	•	70% Training
	•	20% Validation
	•	10% Test


### Data Preprocessing

Images undergo several preprocessing steps to augment the dataset and normalize input data:

1. **Resizing**: Normalizing all images to 512x512 pixels to ensure consistency.
2. **Augmentations**: Enhancing dataset variance through techniques such as random flips, rotations, color adjustments, and applying Gaussian blur.
3. **Normalization**: Scaling pixel values to a predefined range based on dataset statistics to facilitate model convergence.

## Model Architectures

Detailed customizations are made to the following models to suit specific project needs:

### DenseNet121

Modified to have a custom classifier layer replacing the original fully connected layer to predict four classes.

### EfficientNet B1 & B2

These models are similarly adjusted in their final layers to output four disease categories.

## Training Procedures

Models are trained with a focus on handling class imbalance and optimizing generalization:

- **Loss Function**:  Utilization of Cross-Entropy Loss, class-weighted to mitigate class imbalance effects.
- **Optimizer**: Adam optimizer with an initial learning rate of 0.0001, adjusted by ReduceLROnPlateau on validation loss plateau.
- **Schedulers**: Learning rate adjustments using ReduceLROnPlateau for efficient convergence.

Training involves logging detailed metrics for each epoch to monitor progress and adjust parameters dynamically.

## Evaluation and Results

The models undergo rigorous evaluations using precision, recall, and F1-score metrics, supported by detailed error analysis through confusion matrices:

### Performance Metrics

Performance metrics are extensively discussed, emphasizing class-specific insights and overall model effectiveness.

### Visualizations

We apply Grad-CAM to visualize model decision regions on leaf images, providing insights into model focus areas.

## Usage Example

Here's how to load and use the trained model to predict on new leaf images:

\```python
from torchvision import models
import torch

# Load the model
model = models.densenet121(pretrained=False)
model.load_state_dict(torch.load('path_to_model.pth'))

# Prepare the image
from PIL import Image
from torchvision.transforms import transforms
transform = transforms.Compose([...])
image = Image.open('path_to_leaf_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    print(f'Predicted class: {predicted.item()}')
\```

## Contributing

Feel free to fork this project, submit pull requests, or send suggestions to improve the code.

## Citations

Please cite this project as follows:

@misc{your_project_name,
  author = {Your Name},
  title = {Project Title},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your_github}}
}

## License

This project is released under the MIT License. See the LICENSE file for details.
