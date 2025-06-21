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
        input_tensor = tf.constant(image_batch)

        # Check if the model is a KerasLayer (directly callable) or a loaded SavedModel
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            # This is typically a SavedModel loaded by tf.saved_model.load()
            # It should be invoked via its signature.
            try:
                output_dict = model.signatures['serving_default'](input_tensor)
                # The output of a KerasLayer from TF Hub, when saved and loaded,
                # often has a single output. The key might be specific, e.g., 'mobilenet_v2_1_layer'
                # or a generic one like 'output_0'.
                # Let's try to be robust by taking the first value from the output dictionary.
                if not output_dict:
                    raise ValueError("Signature output dictionary is empty.")
                # The actual tensor is the value in the output dictionary.
                # Common key for TF Hub KerasLayer feature vector output is often 'default' or the layer name or 'output_0'.
                # Let's try to get the first output tensor from the dictionary.
                output_key = list(output_dict.keys())[0]
                embedding_tensor = output_dict[output_key]
                
                # Ensure it's a tensor, as some signatures might return other structures
                if not isinstance(embedding_tensor, tf.Tensor):
                    # This case might happen if the signature is more complex.
                    # For MobileNetV2 feature vector, we expect a single tensor.
                    # If it's a different structure, this logic might need adjustment based on model specifics.
                    # Example: if it's a list/tuple of tensors, take the first one.
                    if isinstance(embedding_tensor, (list, tuple)) and len(embedding_tensor) > 0 and isinstance(embedding_tensor[0], tf.Tensor):
                        embedding_tensor = embedding_tensor[0]
                    else:
                        raise ValueError(f"Expected a TensorFlow tensor from the signature output, but got {type(embedding_tensor)} for key '{output_key}'.")

            except Exception as sig_e:
                logger.error(f"Error using 'serving_default' signature for cached MobileNetV2: {sig_e}. Model type: {type(model)}")
                return f"Error processing model signature: {str(sig_e)}"
        
        elif callable(model):
            # This path is for the original hub.KerasLayer object before it's cached,
            # or other directly callable models.
            try:
                embedding_tensor = model(input_tensor) # Direct call for KerasLayer
            except Exception as call_e:
                logger.error(f"Error calling model directly (expected KerasLayer): {call_e}. Model type: {type(model)}")
                return f"Error calling model: {str(call_e)}"
        else:
            # This case should ideally not be reached if model_manager returns a valid model or None
            logger.error(f"Model is not callable and has no 'serving_default' signature. Model type: {type(model)}")
            return "Error: Model is not in a recognized callable or SavedModel format."

        # The output of hub.KerasLayer (and its saved version) is typically (1, num_features)
        # We convert it to a list of floats
        embedding = embedding_tensor.numpy().flatten().tolist()

        return embedding
    except Exception as e:
        # Log the full traceback for better debugging if possible
        logger.error(f"Error during image embedding extraction with MobileNetV2: {e} (model type: {type(model)})", exc_info=True)
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
