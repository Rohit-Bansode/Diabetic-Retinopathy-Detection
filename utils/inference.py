import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


# ── Register custom layers (must match notebook exactly) ─────────────────────

@tf.keras.utils.register_keras_serializable(package="MyModels")
class ReduceMean(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)


@tf.keras.utils.register_keras_serializable(package="MyModels")
class ReduceMax(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)


# ── Custom loss functions (needed to load model) ──────────────────────────────

def categorical_focal_loss(gamma=2.0, alpha=0.25, num_classes=5):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal = alpha * tf.pow(1 - pt, gamma) * ce
        return tf.reduce_mean(focal)
    return loss


def ordinal_mse_loss(num_classes=5):
    def loss(y_true, y_pred):
        class_values = tf.cast(tf.range(num_classes), tf.float32)
        expected_grade = tf.reduce_sum(y_pred * class_values, axis=-1)
        return tf.reduce_mean(tf.square(expected_grade - tf.cast(y_true, tf.float32)))
    return loss


def combined_ordinal_focal_loss(gamma=2.0, alpha=0.25, lam=0.5, num_classes=5):
    focal = categorical_focal_loss(gamma, alpha, num_classes)
    ordinal = ordinal_mse_loss(num_classes)
    def loss(y_true, y_pred):
        return focal(y_true, y_pred) + lam * ordinal(y_true, y_pred)
    return loss


# ── Model load ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load the trained Keras model with all custom objects."""
    custom_objects = {
        "ReduceMean": ReduceMean,
        "ReduceMax": ReduceMax,
        "loss": combined_ordinal_focal_loss(),
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded from {model_path}")
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, processed_img):
    """
    Run inference on a single preprocessed image.
    processed_img: float32 numpy array (300, 300, 3)
    Returns:
        probs  — softmax probabilities (5,)
        grade  — predicted class 0–4
    """
    # EfficientNet expects [0, 255] scaled by its own preprocess_input
    img_input = tf.keras.applications.efficientnet.preprocess_input(
        processed_img.copy()
    )
    img_batch = np.expand_dims(img_input, axis=0)   # (1, 300, 300, 3)

    preds = model.predict(img_batch, verbose=0)      # (1, 5)
    probs = preds[0]                                  # (5,)
    grade = int(np.argmax(probs))
    return probs, grade