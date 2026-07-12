import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


def find_conv_layer(model, layer_name="top_conv"):
    """
    Locate a layer by name, searching nested sub-models too.
    Returns (owning_model, layer) where owning_model is the model object
    that actually contains this layer in its own graph (could be `model`
    itself, or a nested sub-model like the EfficientNetB3 backbone).
    """
    try:
        return model, model.get_layer(layer_name)
    except ValueError:
        pass

    for layer in model.layers:
        if hasattr(layer, "layers"):  # nested sub-model
            try:
                return layer, layer.get_layer(layer_name)
            except ValueError:
                continue

    raise ValueError(
        f"Layer '{layer_name}' not found in model or any nested sub-model."
    )


def make_gradcam_heatmap(model, img_array, last_conv_layer_name="top_conv"):
    """
    Generate Grad-CAM heatmap for a single image.

    Works whether `last_conv_layer_name` lives directly in `model` or inside
    a nested sub-model (e.g. EfficientNetB3 added as a single layer), by
    never constructing a new Model() that spans two different graphs.
    Instead, it runs the ORIGINAL model forward inside a GradientTape and
    captures the conv layer's activation via a temporary output-capturing
    wrapper on that layer's `call`.
    """
    owning_model, conv_layer = find_conv_layer(model, last_conv_layer_name)

    captured = {}
    original_call = conv_layer.call

    def capturing_call(*args, **kwargs):
        out = original_call(*args, **kwargs)
        captured["activation"] = out
        return out

    conv_layer.call = capturing_call

    try:
        img_tensor = tf.convert_to_tensor(img_array)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor, training=False)
            conv_outputs = captured["activation"]
            tape.watch(conv_outputs)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            raise ValueError(
                "Gradients could not be computed w.r.t. the conv layer output. "
                "This usually means the layer's activation isn't actually used "
                "to produce the model's predictions, or gradient watching was "
                "broken by a non-differentiable op between them."
            )
    finally:
        # Always restore the original call, even if something raised above
        conv_layer.call = original_call

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (c,)

    conv_outputs = conv_outputs[0]                          # (h, w, c)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                           # (h, w)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    # Force a clean, contiguous float32 numpy array. TF can hand back
    # float64 (or non-contiguous) arrays depending on backend/version,
    # and cv2.resize raises `func != 0 in function 'cv::hal::resize'`
    # on dtypes it doesn't recognise (notably float64).
    heatmap = heatmap.numpy().astype(np.float32)
    heatmap = np.ascontiguousarray(heatmap)
    return heatmap


def generate_gradcam_overlay(model, processed_img, original_pil_img,
                              last_conv_layer_name="top_conv", alpha=0.4):
    """
    Full Grad-CAM pipeline -> returns PIL Image of overlay.

    processed_img     : float32 numpy (300, 300, 3) - CLAHE output
    original_pil_img  : PIL Image (original upload, any size)
    """
    img_input = tf.keras.applications.efficientnet.preprocess_input(
        processed_img.copy()
    )
    img_batch = np.expand_dims(img_input, axis=0)

    heatmap = make_gradcam_heatmap(model, img_batch, last_conv_layer_name)

    # Belt-and-suspenders: guarantee float32 + contiguous right before resize,
    # even though make_gradcam_heatmap already does this.
    heatmap = np.ascontiguousarray(heatmap.astype(np.float32))

    orig_w, orig_h = original_pil_img.size
    orig_w, orig_h = int(orig_w), int(orig_h)
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colour = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    orig_array = np.array(original_pil_img.convert("RGB")).astype(np.float32)
    overlay = orig_array * (1 - alpha) + heatmap_colour.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)