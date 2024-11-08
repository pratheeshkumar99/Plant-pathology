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
  - [EfficientNetB1](#efficientnetb1)
  - [EfficientNetB2](#efficientnetb2)
  - [Ensemble Methods](#ensemble-methods)
    - [Average Ensemble](#average-ensemble)
    - [Soft Voting](#soft-voting)
- [Training Procedures](#training-procedures)
- [Evaluation and Results](#evaluation-and-results)
- [Visualizations](#visualizations)
  - [Batch Visualization](#batch-visualization)
- [Usage Example](#usage-example)
- [Contributing](#contributing)
- [Citations](#citations)
- [License](#license)

## Dataset

This project utilizes the Kaggle Plant Pathology 2020 - FGVC7 dataset, which comprises various categories of plant leaf images labeled as Healthy, Multiple Diseases, Rust, and Scab. The dataset is designed to reflect real-world conditions and train robust models capable of handling diverse plant conditions.

### Data Splitting

The dataset is split as follows to ensure the model's ability to generalize well on unseen data:
- **70% Training**
- **20% Validation**
- **10% Test**

This stratified split ensures that each set reflects the overall distribution of classes.

#### Class Distribution

The initial class distribution is as follows:
- **Healthy**: Approx. 550 images
- **Multiple Diseases**: Approx. 150 images
- **Rust**: Approx. 600 images
- **Scab**: Approx. 600 images

![Initial Class Distribution](Images/class_distribution.png)

### Data Preprocessing

Images undergo several preprocessing steps to augment the dataset and normalize input data:
1. **Resizing**: Normalizing all images to 512x512 pixels to ensure consistency.
2. **Augmentations**: Enhancing dataset variance through techniques such as random flips, rotations, color adjustments, and applying Gaussian blur.
3. **Normalization**: Scaling pixel values to a predefined range based on dataset statistics to facilitate model convergence.

![Example of Transformed Images](Images/batch_visualization.png)

## Visualizations

### Batch Visualization

To ensure that each training batch is representative of the overall dataset, we monitor the class distribution within batches.

![Class Distribution in One Batch](Images/batch_distribution.png)

## Model Architectures

### Overview of Models Used

The project employs several models and techniques to address the challenges of plant pathology classification:
- DenseNet121
- EfficientNetB1
- EfficientNetB2
- Average Ensemble
- Soft Voting

### DenseNet121

DenseNet121 is a convolutional neural network that is distinct for its dense connectivity pattern. Unlike traditional networks, where each layer is connected only to the next layer, DenseNet connects each layer to every other layer in a feed-forward fashion. This unique architecture helps in alleviating the vanishing-gradient problem, strengthening feature propagation, encouraging feature reuse, and substantially reducing the number of parameters. DenseNet has proven to be effective in various image recognition tasks, particularly in areas where preserving fine details is crucial, such as medical imaging and plant pathology.

![DenseNet Architecture](Images/DenseNet-architecture.png)

For a detailed performance analysis of DenseNet121 on plant pathology classification, see the [DenseNet121 section](#densenet121).

### DenseNet121 Performance Analysis

#### Performance without weighted Loss

##### Overview

This section evaluates the DenseNet121 model's performance without a weighted loss function, establishing a baseline for its capabilities and identifying areas for potential improvement.

##### Training and Validation Loss and Validation Accuracy

The Training and Validation Loss Plot shows how the model's performance changes over time. The training loss (blue line) indicates how well the model fits the training data, while the validation loss (orange line) assesses its generalization to unseen data. The Validation Accuracy Plot provides a crucial metric for assessing model effectiveness under varying training conditions.

![Training and Validation Loss Plot](Images/DenseNet_unweighted_loss_plot.png)
- **Key Observation:** Steady decrease in training loss suggests effective learning, but the plateau in validation loss indicates possible overfitting.
- **Key Observation:** Increasing validation accuracy shows alignment with actual labels, though it plateaus, suggesting a limit to current model configuration benefits.

##### Confusion Matrix

The Confusion Matrix visualizes performance across classes, highlighting accurate classifications and misclassifications.

![Confusion Matrix](Images/DenseNet_unweighted_confusion_plot.png)
- **Key Observation:** High accuracy for most classes but notable misclassifications in 'Multiple Diseases'.

##### Classification Report

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

#### Performance with weighted Loss

##### Introduction

Implementing weighted loss addresses class imbalance by emphasizing minority classes during training, potentially enhancing overall model accuracy and equity.

##### Performance Metrics Post-Weighted Loss

Post-adjustment metrics indicate improved balance and performance across classes.

| Metric     | Value  |
|------------|--------|
| Precision  | 0.8602 |
| Recall     | 0.8623 |
| F1 Score   | 0.8601 |

##### Classification Report Post-Weighted Loss

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

##### Training and Validation Loss Post-Weighted Loss

The plot below shows training and validation loss over epochs, indicating a closer convergence between training and validation loss, suggestive of reduced overfitting.

![Training and Validation Loss Plot](Images/training_plot_weighted_denseNet121.png)

##### Validation Accuracy Plot Post-Weighted Loss

A steady increase in validation accuracy shows the model's improved performance, reaching an optimal point around the 6th epoch.

![Validation Accuracy Plot](Images/validation_accuracy_weighted_denseNet121.png)

##### Updated Confusion Matrix

This matrix reflects improved recognition of under-represented classes and overall accurate predictions.

![Confusion Matrix](Images/Confusion_matrix_weighted_loss.png)







### EfficientNetB1

EfficientNetB1 is a convolutional neural network that forms part of the EfficientNet family, known for balancing depth, width, and resolution through compound scaling. This model optimizes performance and efficiency by using a systematic approach to scale up CNNs, making it more effective without increasing complexity significantly.

EfficientNetB1 incorporates innovations such as MBConv, a mobile inverted bottleneck that utilizes depthwise separable convolutions to reduce computational demand, and squeeze-and-excitation blocks to enhance feature responses across channels. These features help it achieve higher accuracy with fewer parameters compared to other networks of similar size.

This model excels in tasks that require detailed image recognition, making it highly suitable for applications ranging from medical imaging to environmental monitoring, where precision is crucial.

![EfficientNetB1 Architecture](Images/EfficientNetB1.png)

For a detailed performance analysis of DenseNet121 on plant pathology classification, see the [EfficientNetB1 section](#Weighted-EfficientNetB1-Model-Performance-Analysis).

## Weighted EfficientNetB1 Model Performance Analysis

### Introduction

This section reviews the performance of the EfficientNetB1 model after implementing weighted loss to address class imbalances in our dataset. The weighted loss helps to enhance the model's sensitivity towards underrepresented classes, aiming for a more balanced and fair accuracy across all categories.

##### Training and Validation Loss for Weighted EfficientNetB1

The training and validation loss plot shows a substantial decrease in training loss from the initial epochs, which indicates the model is learning effectively from the training data. The validation loss initially decreases and then stabilizes, closely following the training loss, which suggests that the model is not overfitting significantly. This convergence of training and validation loss indicates a good generalization of the model under the current configuration. The convergence between the training and validation loss by the later epochs suggests that the model is well-tuned for the amount of training it received, with possibly minimal gains from additional training without further adjustments or more complex regularization techniques.

![Training and Validation Loss Plot](Images/EfficientNetB1_loss_plot.png)

##### Validation Accuracy for Weighted EfficientNetB1

The validation accuracy plot shows an upward trend, indicating a consistent improvement in model performance on the validation set as training progresses. The model reaches a peak accuracy and maintains it, which is indicative of robust learning outcomes.The stable high accuracy in later epochs highlights the model’s capacity to maintain its performance, suggesting that the learning rate and other hyperparameters are well-set for this phase of training. The slight dips might indicate minor fluctuations in learning but aren’t significant enough to suggest major issues.

![Validation Accuracy Plot](Images/EfficientNetB1_accuracy_plot.png)


##### Classification Report for Weighted EfficientNetB1

The classification report reflects significant improvements in precision, recall, and F1 scores across almost all classes:

| Class            | Precision | Recall | F1 Score | Support |
|------------------|-----------|--------|----------|---------|
| Healthy          | 0.96      | 0.96   | 0.96     | 28      |
| Multiple Diseases| 0.60      | 0.60   | 0.60     | 5       |
| Rust             | 0.97      | 0.97   | 0.97     | 34      |
| Scab             | 0.97      | 0.97   | 0.97     | 32      |
| **Accuracy**     |           |        | 0.95     | 99      |
| **Macro Avg**    | 0.88      | 0.88   | 0.88     | 99      |
| **Weighted Avg** | 0.95      | 0.95   | 0.95     | 99      |

The marked improvement in the Multiple Diseases class from previous metrics indicates that the weighted loss is effective. The high scores in other classes reinforce the model’s capability to accurately classify different plant conditions.

##### Confusion Matrix for Weighted EfficientNetB1

The confusion matrix provides a detailed look at how the model performs across different classes:
- Healthy: Almost perfect prediction with 27 out of 28 correctly predicted.
- Multiple Diseases: This class shows a marked improvement with accurate predictions for 3 out of 5 instances, though there’s still some confusion with other classes, particularly Rust and Scab.
- Rust: High precision and recall with 33 out of 34 correctly predicted, showing that the model is very effective at identifying this condition.
- Scab: Similar to Rust, the model effectively identifies Scab with 31 out of 32 correct predictions.
- Overall Analysis: The confusion matrix reveals high accuracy across most classes with slight confusion between Multiple Diseases and other diseases, suggesting that further model refinement or more targeted data augmentation could help improve these

![Confusion Matrix](Images/EfficientNetB1_confusion_plot.png.png)

