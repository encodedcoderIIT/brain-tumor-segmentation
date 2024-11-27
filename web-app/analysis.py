import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

# Computing Precision
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Computing Sensitivity
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Load the model without the optimizer
model = load_model(
    'web-app/static/model/model_2024_2D_UNet.h5',
    custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'dice_coef_necrotic': dice_coef_necrotic,
        'dice_coef_edema': dice_coef_edema,
        'dice_coef_enhancing': dice_coef_enhancing
    },
    compile=False
)

# Recompile the model with a compatible optimizer
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing])

def analyze(images):
    # Normalize the images
    normalized_images = [(image - np.min(image)) / (np.max(image) - np.min(image)) for image in images]

    # Load and process the images
    slice_idx = normalized_images[0].shape[2] // 2  # Middle slice

    # Resize the images
    IMG_SIZE = 128
    resized_images = [cv2.resize(image[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)) for image in normalized_images]

    # Stack them together to create the multi-channel input
    X_input = np.stack(resized_images, axis=-1)  # Shape will be (IMG_SIZE, IMG_SIZE, len(images))

    # Normalize the input
    X_input = X_input / np.max(X_input)  # Normalizing the input to [0, 1]

    # If you need to add an extra batch dimension
    X_input = np.expand_dims(X_input, axis=0)  # Shape will be (1, IMG_SIZE, IMG_SIZE, len(images))

    # Make a prediction
    prediction = model.predict(X_input)

    # Process the prediction result
    prediction = np.squeeze(prediction)

    # Convert the prediction to a base64 string to return as JSON
    fig, ax = plt.subplots()
    ax.imshow(np.argmax(prediction, axis=-1), cmap='jet')  # Take the channel with the max probability
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return {'prediction': img_base64}


