import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.preprocessing.image import img_to_array

class Model:

    def __init__(self):
        """
        Initialize the internal state of the model. Note that the __init__
        method cannot accept any arguments.

        The following is an example loading the weights of a pre-trained
        model.
        """
        self.neural_network = [
            tfk.models.load_model("BloodCells_EfficientNetV2S_FT_eff_99.16.keras"),
            tfk.models.load_model("BloodCells_EfficientNetV2S_FT_TensorFlowAug_98.48.keras")
        ]


    def add_gaussian_noise(self, image, mean=0.0, stddev=0.05):
        noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=image.dtype)
        return image + noise


    def tta_augmentations(self, image):
        """
        Apply domain-specific augmentations for TTA.
        Args:
            image: TensorFlow image tensor (H, W, C).
        Returns:
            List of augmented images.
        """
        augmentations = [
            image,  # Original
            tf.image.flip_left_right(image),  # Horizontal flip
            tf.image.flip_up_down(image),  # Vertical flip
            tf.image.rot90(image, k=1),  # Rotate 90 degrees
            tf.image.adjust_brightness(image, delta=0.1),  # Brightness adjustment
            tf.image.adjust_contrast(image, contrast_factor=1.1),  # Contrast adjustment
            tf.image.adjust_saturation(image, saturation_factor=1.2),  # Saturation adjustment
            self.add_gaussian_noise(image, mean=0.0, stddev=0.05),  # Gaussian noise
        ]
        return augmentations
    

    def tta_predict_with_weights(self, image, weights):
        """
        Perform TTA with optimized augmentations.
        Args:
            model: Trained TensorFlow/Keras model.
            image: Original image (PIL or Tensor).
            augmentations: Augmentation function.
            image_size: Tuple (height, width) for resizing.
        Returns:
            Aggregated probabilities as the final prediction.
        """
        # remove first dimension if present
        if image.shape[0] == 1:
            image = image[0]

        # Preprocess the image
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(img_to_array(image))  # Convert to tensor if needed
        
        # Generate augmented images
        augmented_images = self.tta_augmentations(image)
        augmented_images = tf.stack(augmented_images)

        weights = weights * len(self.neural_network)
        predictions = []

        # Predict the label using multiple models for each augmented image
        for model in self.neural_network:
            predictions.extend(model.predict(np.array(augmented_images)))
        predictions = tf.nn.softmax(predictions, axis=-1).numpy()
        return self.weighted_aggregation(predictions, weights)
    

    def weighted_aggregation(self, predictions, weights):
        """
        Perform weighted aggregation of predictions.
        Args:
            predictions: List of prediction arrays.
            weights: List of weights corresponding to each augmentation.
        Returns:
            Weighted average of predictions.
        """
        weighted_preds = [pred * weight for pred, weight in zip(predictions, weights)]
        return np.sum(weighted_preds, axis=0) / np.sum(weights)


    def predict(self, X):
        """
        Predict the labels corresponding to the input X. Note that X is a numpy
        array of shape (n_samples, 96, 96, 3) and the output should be a numpy
        array of shape (n_samples,). Therefore, outputs must no be one-hot
        encoded.

        The following is an example of a prediction from the pre-trained model
        loaded in the __init__ method.
        """
        # Resize all images in X to 96x96x3 before prediction
        X = np.array([tf.image.resize_with_pad(img, 96, 96).numpy().astype(np.uint8) for img in X])

        final_preds = []
        for i in range(X.shape[0]):
            sample = np.expand_dims(X[i], axis=0)
            preds = self.tta_predict_with_weights(
                sample, 
                weights = [1.2, 0.85, 0.9, 0.75, 1, 1, 1, 1.1],
                image_size=(96, 96)
            )
            preds = np.argmax(preds)
            final_preds.append(preds)
        return np.array(final_preds)