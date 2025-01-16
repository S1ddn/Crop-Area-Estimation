# -*- coding: utf-8 -*-
"""yolov8_grid1-6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11QRFbjoSp9Fg2LSfwv3vWC9rdtCaaDJA
"""

!pip install ultralytics
!pip install rasterio tensorflow

from google.colab import drive
drive.mount('/content/drive')

import rasterio
import tensorflow as tf

from ultralytics import YOLO
import os
import shutil
from sklearn.model_selection import train_test_split
import random

import os
from random import choice

#arrays to store file names
imgs =[]
xmls =[]

#setup dir names
trainPath =('/content/drive/MyDrive/data-code/yolo_1-6_train-label/images/train')
valPath = ('/content/drive/MyDrive/data-code/yolo_1-6_train-label/images/val')
crsPath = ('/content/drive/MyDrive/grid1-6_plant') #dir where images and annotations stored

#setup ratio (val ratio = rest of the files in origin dir after splitting into train and test)
train_ratio = 0.8
val_ratio = 0.2


#total count of imgs
totalImgCount = len(os.listdir(crsPath))/2

#soring files to corresponding arrays
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:
        if filename.endswith('.txt'):
            xmls.append(filename)
        else:
            imgs.append(filename)


#counting range for cycles
countForTrain = int(len(imgs)*train_ratio)
countForVal = int(len(imgs)*val_ratio)
print("Total number of images: ", len(imgs))
print("training images: ",countForTrain)
print("Validation images: ",countForVal)

import shutil, sys
trainimagePath = ('/content/drive/MyDrive/data-code/yolo_1-6_train-label_29/images/train')
trainlabelPath = ('/content/drive/MyDrive/data-code/yolo_1-6_train-label_29/labels/train')
valimagePath = ('/content/drive/MyDrive/data-code/yolo_1-6_train-label_29/images/val')
vallabelPath = ('/content/drive/MyDrive/data-code/yolo_1-6_train-label_29/labels/val')
#cycle for train dir
for x in range(countForTrain):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file

    #move both files into train dir
    #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
    #shutil.move(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))
    shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
    shutil.copy(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))


    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)



#cycle for test dir
for x in range(countForVal):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file

    #move both files into train dir
    #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
    #shutil.move(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))
    shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
    shutil.copy(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))

    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)

#rest of files will be validation files, so rename origin dir to val dir
#os.rename(crsPath, valPath)
#shutil.move(crsPath, valPath)

"""** **bold text**TRAINING WITH YOLOV8N**"""

# Define the model
model = YOLO("yolov8n.pt")

# Use the model
model.train(data='/content/drive/MyDrive/data-code/data29.yaml', epochs=5)

"""**Validation with YOLOV8N**"""

model.val(save_json=True)

"""**Results of YOLOV8N**"""

from IPython.display import Image, display
import glob

# Validate the model and specify the output directory (optional)
save_dir = '/content/drive/MyDrive/yolo1-6_validation_output'
metrics = model.val(save_dir=save_dir)

# List all images in the validation output directory
validation_images = glob.glob(f"{save_dir}/*.jpg")

# Display each image from the validation results
for img_path in validation_images:
    display(Image(filename=img_path))

import glob
from IPython.display import Image, display

# Directory where the results are saved
result_dir = 'runs/detect/train4'

# List all .jpg images in the result directory
result_images = glob.glob(f"{result_dir}/*.jpg")

# Check if any images are found and display them
if result_images:
    for img_path in result_images:
        display(Image(filename=img_path))
else:
    print("No result images found in the specified directory.")

"""**Prediction with YOLOV8N**"""

# NEW code
from PIL import Image
model = YOLO('/content/drive/MyDrive/data-code/yolo8-runs-v8n/detect/train/weights/best.pt')
# Load the image using PIL
image_path = '/content/drive/MyDrive/grid1-6_plant/Copy of subimage_3_55.png'
image = Image.open(image_path)

# Perform prediction on the image
results = model.predict(source=image)

# Extract the first result (since we are predicting on one image)
result = results[0]

# Extract bounding boxes, confidence scores, and class predictions
boxes = result.boxes.xyxy  # Coordinates of the bounding boxes (x1, y1, x2, y2)
confidences = result.boxes.conf  # Confidence scores for each box
classes = result.boxes.cls  # Class indices for each detected object

# Print the extracted information
print(f"Bounding Boxes: {boxes}")
print(f"Confidence Scores: {confidences}")
print(f"Classes: {classes}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Convert the PIL image to a NumPy array for plotting
image_np = np.array(image)

# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image_np)

# Loop over the detected boxes and plot them
for i in range(len(boxes)):
    box = boxes[i].tolist()  # Convert to list for easy handling
    confidence = confidences[i].item()  # Get confidence score
    class_idx = int(classes[i].item())  # Get class index

    # Extract the box coordinates
    x1, y1, x2, y2 = box

    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the axes
    ax.add_patch(rect)

    # Add label and confidence score
    ax.text(x1, y1 - 10, f'Class: {class_idx}, Conf: {confidence:.2f}', color='white', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))

# Show the image with bounding boxes
plt.axis('off')  # Hide axes
plt.show()

"""**TRANINIG WITH YOLOV8S**"""

import os
from ultralytics import YOLO

# Define the model with the YOLOv8s weights
model = YOLO("yolov8s.pt")

# Define the data configuration file path and project directory in Google Drive
data_path = "/content/drive/MyDrive/data-code/data29.yaml"
project_dir = "/content/drive/MyDrive/yolo_runs"

# Train the model with specified dataset and parameters
model.train(data=data_path, epochs=5, project=project_dir, name='exp1')

save_dir = "/content/drive/MyDrive/yolo_models"
os.makedirs(save_dir, exist_ok=True)

# Optionally save a specific version of the model
model.save("/content/drive/MyDrive/yolo_models/yolo_model_1.pt")

"""**VALIDATING WITH YOLOV8S**"""

model.val()

import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt


# Directories for validation images and corresponding labels
val_images_path = '/content/drive/MyDrive/data-code/yolo_1-6_train-label/images/val'
val_labels_path = '/content/drive/MyDrive/data-code/yolo_1-6_train-label/labels/val'

# Lists to hold true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through each validation image
for filename in os.listdir(val_images_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust based on your image types
        image_path = os.path.join(val_images_path, filename)

        # Load the image
        image = Image.open(image_path)

        # Perform prediction
        results = model.predict(source=image)
        result = results[0]

        # Extract predicted classes
        predicted_classes = result.boxes.cls.tolist()

        # Assuming labels are in YOLO format and correspond to filenames
        label_file = os.path.join(val_labels_path, filename.replace('.png', '.txt').replace('.jpg', '.txt'))

        # Check if the label file exists before reading
        if os.path.exists(label_file):
            # Read true labels from label file
            with open(label_file, 'r') as f:
                for line in f:
                    # Assuming label format: <class_id> <x_center> <y_center> <width> <height>
                    class_id = int(line.split()[0])
                    true_labels.append(class_id)

        # Append predicted classes
        predicted_labels.extend(predicted_classes)

# Convert to NumPy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Debug: Check the contents of true_labels and predicted_labels
print("True Labels:", true_labels)
print("Predicted Labels:", predicted_labels)

# Check if there are any labels collected before proceeding
if len(true_labels) == 0 or len(predicted_labels) == 0:
    print("No labels collected. Please check your validation dataset.")
else:
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_labels))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

"""**PREDICTION**"""

# NEW code
from PIL import Image
model = YOLO('/content/drive/MyDrive/data-code/yolo8-runs-v8s/detect/train/weights/best.pt')
# Load the image using PIL
image_path = '/content/drive/MyDrive/grid1-6_plant/Copy of subimage_3_55.png'
image = Image.open(image_path)

# Perform prediction on the image
results = model.predict(source=image)

# Extract the first result (since we are predicting on one image)
result = results[0]

# Extract bounding boxes, confidence scores, and class predictions
boxes = result.boxes.xyxy  # Coordinates of the bounding boxes (x1, y1, x2, y2)
confidences = result.boxes.conf  # Confidence scores for each box
classes = result.boxes.cls  # Class indices for each detected object

# Print the extracted information
print(f"Bounding Boxes: {boxes}")
print(f"Confidence Scores: {confidences}")
print(f"Classes: {classes}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Convert the PIL image to a NumPy array for plotting
image_np = np.array(image)

# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image_np)

# Loop over the detected boxes and plot them
for i in range(len(boxes)):
    box = boxes[i].tolist()  # Convert to list for easy handling
    confidence = confidences[i].item()  # Get confidence score
    class_idx = int(classes[i].item())  # Get class index

    # Extract the box coordinates
    x1, y1, x2, y2 = box

    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the axes
    ax.add_patch(rect)

    # Add label and confidence score
    ax.text(x1, y1 - 10, f'Class: {class_idx}, Conf: {confidence:.2f}', color='white', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))

# Show the image with bounding boxes
plt.axis('off')  # Hide axes
plt.show()