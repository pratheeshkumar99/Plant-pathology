
# Plant Pathology Classification Using Deep Learning

## Project Overview
This repository contains a deep learning project aimed at classifying plant diseases from images of leaves. The project utilizes DenseNet121 and EfficientNet models, incorporating modifications and ensemble techniques to optimize performance for plant pathology diagnostics.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [Overview of Models Used](#overview-of-models-used)
  - [DenseNet121](#densenet121)
    - [Performance Without Weighted Loss](#performance-without-weighted-loss)
    - [Performance With Weighted Loss](#performance-with-weighted-loss)



## Dataset

This project utilizes a kagglePlant Pathology 2020 - FGVC7 comprising various categories of plant leaf images labeled as Healthy, Multiple Diseases, Rust, and Scab. The dataset aims to reflect real-world conditions and is designed to train robust models capable of handling diverse plant conditions

### Data Splitting

The dataset is split as follows to ensure the model’s ability to generalize well on unseen data:
- **70% Training**
- **20% Validation**
- **10% Test**

This stratified split ensures that each set reflects the overall distribution of classes as shown in the initial class distribution.

#### Class Distribution
The initial class distribution is as follows:
- **Healthy**: Approx. 550 images
- **Multiple Diseases**: Approx. 150 images
- **Rust**: Approx. 600 images
- **Scab**: Approx. 600 images

![Initial Class Distribution](Images/class_distribution.png)

The class distribution across training, validation, and test sets is maintained proportionally, as visualized in the following charts:

![Distribution after Splitting the data](Images/splitDistribution.png)
### Data Preprocessing

Images undergo several preprocessing steps to augment the dataset and normalize input data:

1. **Resizing**: Normalizing all images to 512x512 pixels to ensure consistency.
2. **Augmentations**: Enhancing dataset variance through techniques such as random flips, rotations, color adjustments, and applying Gaussian blur. Below are examples of how these transformations affect the images:
3. **Normalization**: Scaling pixel values to a predefined range based on dataset statistics to facilitate model convergence.

![Example of Transformed Images](Images/batch_visualization.png)

## Visualizations

### Batch Visualization
To ensure that each training batch is representative of the overall dataset, we monitor the class distribution within batches. The typical distribution within a single batch is visualized as follows:

![Class Distribution in One Batch](Images/batch_distribution.png)

## DenseNet121 Performance Analysis

### Performance Before Weighted Loss

#### Overview
This section evaluates the DenseNet121 model's performance without a weighted loss function, establishing a baseline for its capabilities and identifying areas for potential improvement.

#### Training and Validation Loss
The Training and Validation Loss Plot shows how the model's performance changes over time. The training loss (blue line) indicates how well the model fits the training data, while the validation loss (orange line) assesses its generalization to unseen data.

![Training and Validation Loss Plot](Images/DenseNet_unweighted_loss_plot.png)
- **Key Observation:** Steady decrease in training loss suggests effective learning, but the plateau in validation loss indicates possible overfitting.

#### Validation Accuracy
The Validation Accuracy Plot provides a crucial metric for assessing model effectiveness under varying training conditions.

![Validation Accuracy Plot](Images/DenseNet_unweighted_accuracy_plot.png)
- **Key Observation:** Increasing validation accuracy shows alignment with actual labels, though it plateaus, suggesting a limit to current model configuration benefits.

#### Confusion Matrix
The Confusion Matrix visualizes performance across classes, highlighting accurate classifications and misclassifications.

![Confusion Matrix](Images/DenseNet_unweighted_confusion_plot.png)
- **Key Observation:** High accuracy for most classes but notable misclassifications in 'Multiple Diseases'.

#### Classification Report
The table below summarizes the precision, recall, and F1 scores for each class before applying weighted loss.

| Class               | Precision | Recall | F1 Score | Support |
|---------------------|-----------|--------|----------|---------|
| Healthy             | 0.90      | 0.96   | 0.93     | 28      |
| Multiple Diseases   | 0.50      | 0.20   | 0.29     | 5       |
| Rust                | 0.97      | 0.94   | 0.96     | 34      |
| Scab                | 0.88      | 0.94   | 0.91     | 32      |
| **Accuracy**        |           |        | 0.91     | 99      |
| **Macro Avg**       | 0.81      | 0.76   | 0.77     | 99      |
| **Weighted Avg**    | 0.90      | 0.91   | 0.90     | 99      |

---

### Performance with Weighted Loss

#### Introduction
Implementing weighted loss addresses class imbalance by emphasizing minority classes during training, potentially enhancing overall model accuracy and equity.

#### Performance Metrics Post-Weighted Loss
Post-adjustment metrics indicate improved balance and performance across classes.

| Metric     | Value  |
|------------|--------|
| Precision  | 0.8602 |
| Recall     | 0.8623 |
| F1 Score   | 0.8601 |

#### Classification Report Post-Weighted Loss
Detailed performance per class after applying weighted loss highlights improvements.

| Class               | Precision | Recall | F1 Score | Support |
|---------------------|-----------|--------|----------|---------|
| Healthy             | 0.90      | 1.00   | 0.95     | 28      |
| Multiple Diseases   | 0.60      | 0.60   | 0.60     | 5       |
| Rust                | 1.00      | 0.91   | 0.95     | 34      |
| Scab                | 0.94      | 0.94   | 0.94     | 32      |
| **Accuracy**        |           |        | 0.93     | 99      |
| **Macro Avg**       | 0.86      | 0.86   | 0.86     | 99      |
| **Weighted Avg**    | 0.93      | 0.93   | 0.93     | 99      |

#### Training and Validation Loss Post-Weighted Loss
The plot below shows training and validation loss over epochs, indicating a closer convergence between training and validation loss, suggestive of reduced overfitting.

![Training and Validation Loss Plot](Images/training_plot_weighted_denseNet121.png)

#### Validation Accuracy Plot Post-Weighted Loss
A steady increase in validation accuracy shows the model's improved performance, reaching an optimal point around the 6th epoch.

![Validation Accuracy Plot](Images/validation_accuracy_weighted_denseNet121.png)

#### Updated Confusion Matrix
This matrix reflects improved recognition of under-represented classes and overall accurate predictions.

![Confusion Matrix](Images/Confusion_matrix_weighted_loss.png)

<!-- ## Model Architectures

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

This project is released under the MIT License. See the LICENSE file for details. -->
