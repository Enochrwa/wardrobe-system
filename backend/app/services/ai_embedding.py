# This module provides functions for extracting image embeddings using lightweight models.
# It uses MobileNetV2 from TensorFlow Hub for efficient embedding generation.
# Enhanced with persistent caching to prevent re-downloading models.

import tensorflow as tf
from PIL import Image
import numpy as np
from typing import List, Optional, Union
import logging
from .model_manager import get_model_manager

logger = logging.getLogger(__name__)

def get_image_embedding(image: Image.Image) -> Union[List[float], str]:
    """
    Extracts image embedding using MobileNetV2 with persistent caching.

    Args:
        image: A PIL Image object.

    Returns:
        A list of floats representing the image embedding, or a string with an error message.
    """
    model_manager = get_model_manager()
    
    # Get the model (from cache or download)
    model = model_manager.get_model("mobilenet_v2")
    
    if model is None:
        return "Error: Image embedding model (MobileNetV2) could not be loaded. Cannot extract embedding."

    try:
        # Preprocess the image
        # 1. Convert to RGB if it's not
        image = image.convert("RGB")
        # 2. Resize to MobileNetV2's expected input size (224x224)
        image_resized = image.resize((224, 224))
        # 3. Convert to numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image_resized) / 255.0
        # 4. Add a batch dimension (model expects batch_size, height, width, channels)
        image_batch = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Generate embedding
        # Check if the model is a KerasLayer (directly callable) or a loaded SavedModel
        if hasattr(model, 'signatures'): # Characteristic of a loaded SavedModel
            # Use the serving_default signature
            # Convert image_batch to a TensorFlow constant for the signature
            input_tensor = tf.constant(image_batch)
            try:
                # Common output key for KerasLayer models from TF Hub when saved/loaded
                # is often 'output_0' or the name of the layer.
                # For feature vector models, it might also be 'feature_vector' or 'default'.
                # Let's try to access it dynamically or assume a common one.
                # The MobileNetV2 feature vector layer from TF Hub might output a dict.
                output_dict = model.signatures['serving_default'](input_tensor)
                # The actual tensor can be under various keys, often 'output_0' for simple Keras layers,
                # or a more descriptive name like 'feature_vector'.
                # Let's inspect the keys if possible, or try a common default.
                # A common output for TF Hub feature vector KerasLayer is a single tensor,
                # which when served might be keyed by 'output_0' or a specific name.
                # If it's directly the KerasLayer, this path isn't taken.
                # If it's the loaded SavedModel from hub.KerasLayer, the output structure
                # should ideally match. The KerasLayer itself outputs a tensor directly, not a dict.
                # hub.load() on a feature vector might give a dict.
                # tf.saved_model.load() on a KerasLayer model should give something that can be called.
                # The issue is that `model(image_batch)` fails.
                # Let's assume the signature returns a dictionary and we need to find the embedding tensor.
                # Common key for TF Hub KerasLayer feature vector output is often 'default' or the layer name.
                # If the original KerasLayer was model(inputs), the output is a tensor.
                # If tf.saved_model.load wraps it, the signature might return a dict.
                # Let's try to get the first output from the dictionary.
                output_key = list(output_dict.keys())[0]
                embedding_tensor = output_dict[output_key]
            except KeyError:
                # Fallback if 'serving_default' or the output key is not as expected
                # This might indicate a different model structure or an error in saving/loading
                logger.error("Failed to get embedding using 'serving_default' signature and expected output key.")
                # Try calling the model directly if it's a restored Keras model that somehow became callable
                # This is less likely if the initial error was '_UserObject is not callable'
                try:
                    embedding_tensor = model(input_tensor) # Try direct call if signature fails
                except Exception as direct_call_e:
                    logger.error(f"Direct call also failed after signature attempt: {direct_call_e}")
                    raise direct_call_e # Re-raise the error that led here

        else: # Assume it's directly callable (e.g., original KerasLayer not from cache)
            embedding_tensor = model(image_batch)


        # The output of hub.KerasLayer is already a tensor, often (1, num_features)
        # We convert it to a list of floats
        embedding = embedding_tensor.numpy().flatten().tolist()

        return embedding
    except Exception as e:
        logger.error(f"Error during image embedding extraction with MobileNetV2: {e} (model type: {type(model)})")
        return f"Error during image embedding extraction: {str(e)}"

# Example usage (optional, can be removed or commented out)
if __name__ == '__main__':
    # This part is for testing the module directly.
    # It requires Pillow and TensorFlow to be installed.
    # Create a dummy black image for testing
    try:
        img = Image.new('RGB', (224, 224), color = 'black')
        logger.info("Attempting to extract embedding for a dummy image...")
        
        embedding_result = get_image_embedding(img)
        if isinstance(embedding_result, list):
            logger.info(f"Successfully extracted embedding (first 10 features): {embedding_result[:10]}")
            logger.info(f"Embedding dimension: {len(embedding_result)}")
        else:
            logger.error(f"Embedding extraction failed: {embedding_result}")

    except Exception as e:
        logger.error(f"Error in example usage: {e}")
