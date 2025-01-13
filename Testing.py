import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the TensorFlow model from the SavedModel directory
def load_model(model_path):
    return tf.saved_model.load(model_path)

# Image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB") 
    image = image.resize((256, 256)) 
    image = (np.array(image) / 255.0).astype(np.float32) 
    image = np.expand_dims(image, axis=0)  
    return image

# Function to perform gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255
    return Image.fromarray(np.uint8(table[image]))

# Function to perform dehazing and visualize results
def test_image(model, img_path):
    # Load and preprocess the image
    original_image = Image.open(img_path).convert('RGB')
    input_image = preprocess_image(img_path)

    # Perform prediction using the model
    infer = model.signatures["serving_default"]
    predictions = infer(tf.convert_to_tensor(input_image, dtype=tf.float32))
    
    # Extract the output tensor (assuming single output)
    enhanced_image = predictions[next(iter(predictions))].numpy()
    enhanced_image = (enhanced_image[0] * 255).astype(np.uint8)  # Scale to [0, 255] without clipping
    enhanced_image = Image.fromarray(enhanced_image)

    # Apply gamma correction if the image is too dark
    enhanced_image = adjust_gamma(np.array(enhanced_image), gamma=1.2)  # Try gamma > 1 to brighten

    # Plot original and enhanced images side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(enhanced_image)
    ax[1].set_title("Dehazed Image")
    ax[1].axis("off")
    
    plt.show()

# Path to the saved model directory and test image
model_path = "C:\\Users\\sraja\\OneDrive\\Desktop\\Project\\model_path"
test_image_path = "C:\\Users\\sraja\\Downloads\\images.jpeg"

# Load the model and test an image
model = load_model(model_path)
test_image(model, test_image_path)
