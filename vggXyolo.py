import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import LeakyReLU
from sklearn.metrics import confusion_matrix
import rasterio
import matplotlib.patches as patches
import random
from ultralytics import YOLO

warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

# Helper functions

def load_and_resize_image(path, size=(32, 32)):
    img = Image.open(path)
    img = img.resize(size)
    img_arr = np.array(img).reshape(*size, 3) / 255.0
    return img_arr

def preprocess_data(csv_path, img_column='Path', label_column='Label'):
    data = pd.read_csv(csv_path)
    data[img_column] = data[img_column].apply(load_and_resize_image)
    X = np.stack(data[img_column])
    y = to_categorical(data[label_column], num_classes=3)
    return X, y

def save_predictions_to_csv(predictions, csv_path, labels):
    predicted_labels = [np.argmax(pred) for pred in predictions]
    df = pd.DataFrame({'File Path': labels, 'Predicted Label': predicted_labels})
    df.to_csv(csv_path, index=False)
    return df

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Paths
train_csv = '/content/drive/MyDrive/data-code/estimation_train_labels.csv'
test_csv = '/content/drive/MyDrive/testing_images_crop_area_est.csv'
kml_output_path = '/content/drive/MyDrive/Grid_Predictions.kml'

# Data Preprocessing
X_train, y_train = preprocess_data(train_csv)
X_test, y_test = preprocess_data(test_csv)

# VGG16 Model
def build_vgg16_model(weights_path):
    pretrained_model = VGG16(input_shape=(32, 32, 3), include_top=False, weights=None)
    pretrained_model.load_weights(weights_path)
    for layer in pretrained_model.layers:
        layer.trainable = False
    x = Flatten()(pretrained_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(pretrained_model.input, x)
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    return model

vgg_weights_path = '/content/drive/MyDrive/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_vgg = build_vgg16_model(vgg_weights_path)

# Training
history = model_vgg.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictions and Confusion Matrix
y_pred = model_vgg.predict(X_test)
plot_confusion_matrix(y_test, y_pred, ['NotPlantation', 'Plantation', 'TreeCover'])

# YOLO Integration
def predict_with_yolo(model_path, subimages):
    yolo_model = YOLO(model_path)
    predictions = []
    for img_path in subimages:
        img = Image.open(img_path)
        results = yolo_model.predict(img, conf=0.25)
        if results and results[0].boxes.cls.shape[0] > 0:
            pred_class = yolo_model.names[int(results[0].boxes.cls[0])]
        else:
            pred_class = 'No Detection'
        predictions.append({'Path': img_path, 'YOLO_Label': pred_class})
    return pd.DataFrame(predictions)

# Example prediction with YOLO
plantation_images = pd.read_csv('/content/drive/MyDrive/subimage_paths.csv')
plantation_images = plantation_images[plantation_images['Predicted Label'] == 1]['File Path'].tolist()
yolo_results = predict_with_yolo('/content/drive/MyDrive/yolo_models/yolo_model_1.pt', plantation_images)
yolo_results.to_csv('/content/drive/MyDrive/yolo_classified_plantation_subimages.csv', index=False)

# Save Model
model_vgg.save('/content/drive/MyDrive/model_vgg.h5')
