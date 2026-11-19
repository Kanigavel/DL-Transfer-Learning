# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement 
Image classification is a fundamental task in computer vision where images are categorized into predefined classes. The objective of this project is to develop an image classification model using transfer learning with the pre-trained VGG19 architecture, adapting its learned features to the given dataset, and to evaluate the model’s performance using appropriate metrics.


## Neural Network Model
<img width="213" height="236" alt="image" src="https://github.com/user-attachments/assets/e674e666-4eac-4106-a6b2-142cac67cc63" />


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2:

Load training and testing datasets using ImageFolder.

### STEP 3:

Visualize sample images from the dataset.

### STEP 4:

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5:

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6:

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.







## PROGRAM

### Name: Kanigavel M

### Register Number:212224240070

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```
```
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
```
```
# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "C:\\Time series\\DL\\chip_data\\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
```
```
# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
```
```
# Show sample images from the training dataset
print("Kanigavel M")
print("212224240070")
show_sample_images(train_dataset)
```
```
# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")
print(f"Total number of testing samples: {len(test_dataset)}")
# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
```
```
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

```
```
num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)
# Load Pretrained Model
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace final classifier layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (train only last layer)
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

```
```
# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
```
```
from torchsummary import summary
# Print model summary
print("kanigavel M")
print("212224240070")
print("Model Summary:")
summary(model, input_size=(3, 224, 224))
```
```
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    
    # Plot training and validation loss
   
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```
```
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
```
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

```
```
train_model(model, train_loader,test_loader)
test_model(model, test_loader)
```
```
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _,pred=torch.max(output,1)
        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(pred)
        predicted = (prob > 0.5).int().item()

    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    print("Kanigavel M")
    print("212224240070")
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')
```
```
# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)
```

```
#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```
### OUTPUT
## Sample image
<img width="1138" height="303" alt="image" src="https://github.com/user-attachments/assets/dda02fb2-4730-4223-891f-173a29d33d01" />

## Number of Training Samples
<img width="644" height="121" alt="image" src="https://github.com/user-attachments/assets/a3732ad1-a361-4de1-b920-6b0c734e86c9" />

## Number of Classes
<img width="619" height="353" alt="image" src="https://github.com/user-attachments/assets/30527668-d266-4286-bfc9-0983fd71702c" />

## Model Summary 
<img width="700" height="519" alt="image" src="https://github.com/user-attachments/assets/b46e7cc7-2836-40ac-94e4-269a3eac2eec" />

## Training Loss, Validation Loss Vs Iteration Plot

<img width="530" height="170" alt="image" src="https://github.com/user-attachments/assets/8ca3f887-9695-4462-8189-f45e0776f002" />
<img width="788" height="498" alt="image" src="https://github.com/user-attachments/assets/ba90dccb-ab09-4018-a2c3-932d4fb827a3" />



## Confusion Matrix
<img width="670" height="513" alt="image" src="https://github.com/user-attachments/assets/3f8cf748-739d-4c02-8013-fd5b7c36424e" />

## Classification Report

<img width="525" height="190" alt="image" src="https://github.com/user-attachments/assets/004c0987-e77e-4289-85a7-da0440e41852" />


### New Sample Data Prediction
<img width="544" height="438" alt="image" src="https://github.com/user-attachments/assets/ebc2d11f-f6ca-45d9-828e-6ec92095fd07" />

<img width="486" height="444" alt="image" src="https://github.com/user-attachments/assets/6626f5ef-f872-4025-b94a-06eee63a2914" />

## RESULT
Thus, an image classification model was developed using transfer learning with the VGG19 architecture for the given dataset, achieving limited classification performance.
