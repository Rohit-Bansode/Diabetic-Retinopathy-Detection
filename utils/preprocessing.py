import cv2
import numpy as np

IMG_SIZE = 300

def crop_image_from_gray(img, tol=7):
    """Remove black borders common in fundus images."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    check_shape = img[np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        return img
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def green_channel_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to green channel of RGB image."""
    img = img.astype(np.uint8)
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g_enhanced = clahe.apply(g)
    return cv2.merge((r, g_enhanced, b))


def preprocess_image(img_array):
    """
    Full preprocessing pipeline:
    RGB numpy array -> crop borders -> resize -> green-channel CLAHE
    Returns float32 array of shape (300, 300, 3)
    """
    img = img_array.astype(np.uint8)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = green_channel_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8))
    return img.astype(np.float32)