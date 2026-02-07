import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Kernel builders
# -----------------------------
def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel (ksize x ksize)."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def box_kernel_2d(ksize: int) -> np.ndarray:
    """Create a normalized 2D box (mean) kernel (ksize x ksize)."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    k = np.ones((ksize, ksize), dtype=np.float32)
    k /= np.sum(k)
    return k


# -----------------------------
# Spatial convolution
# -----------------------------
def blur_spatial(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Spatial-domain filtering via convolution.
    BORDER_REFLECT is a good boundary choice.
    """
    return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)


# -----------------------------
# Frequency domain helpers
# -----------------------------
def psf_to_otf(psf: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """
    Convert PSF (spatial kernel) to OTF (frequency response) by:
    - zero-padding to out_shape
    - circularly shifting kernel center to (0,0)
    """
    H, W = out_shape
    kh, kw = psf.shape

    otf = np.zeros((H, W), dtype=np.float32)
    otf[:kh, :kw] = psf.astype(np.float32)

    cy, cx = kh // 2, kw // 2
    otf = np.roll(otf, shift=-cy, axis=0)
    otf = np.roll(otf, shift=-cx, axis=1)

    return otf


def blur_frequency(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Frequency-domain filtering:
      convolution in space  <=>  multiplication in frequency

    We reflect-pad the image to approximate linear convolution and match
    the spatial filter's boundary behavior, then crop back.
    """
    if img.ndim != 2:
        raise ValueError("Pass a single-channel (grayscale) image.")

    H, W = img.shape
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2

    # Reflect pad so borders behave similar to cv2.BORDER_REFLECT
    img_pad = cv2.copyMakeBorder(
        img, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT
    ).astype(np.float32)

    Hp, Wp = img_pad.shape

    # PSF -> OTF
    otf = psf_to_otf(kernel, (Hp, Wp))

    # FFTs
    F = np.fft.fft2(img_pad)
    Hf = np.fft.fft2(otf)

    # Multiply in frequency domain
    G = F * Hf

    # Back to spatial domain
    g = np.fft.ifft2(G).real

    # Crop to original size
    return g[pad_y:pad_y + H, pad_x:pad_x + W]


# -----------------------------
# Error metrics
# -----------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


# -----------------------------
# Main demo
# -----------------------------
def main():
    # ---- Load image (grayscale) ----
    image_path = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/a4_calib_images2/cameraman.png"  # <- change if needed
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Could not read '{image_path}'. Put it in the same folder or update image_path."
        )
    img_f = img.astype(np.float32)

    # ---- Choose filter ----
    ksize = 31   # must be odd
    sigma = 5.0
    kernel = gaussian_kernel_2d(ksize, sigma)
    # kernel = box_kernel_2d(ksize)  # alternative

    # ---- Spatial convolution ----
    spatial = blur_spatial(img_f, kernel)

    # ---- Frequency multiplication ----
    freq = blur_frequency(img_f, kernel)

    # ---- Compare results ----
    error_mse = mse(spatial, freq)
    error_rmse = rmse(spatial, freq)
    diff = np.abs(spatial - freq)

    print("==== Verification: Convolution in Space == Multiplication in Frequency ====")
    print(f"Kernel: {ksize}x{ksize}, sigma={sigma}")
    print(f"MSE  (spatial, frequency) = {error_mse:.8e}")
    print(f"RMSE (spatial, frequency) = {error_rmse:.8e}")
    print(f"Max |difference|          = {float(np.max(diff)):.6e}")

    # ---- Frequency spectra for visualization (DISPLAY ONLY) ----
    # Pad same way as blur_frequency so the spectra correspond to what was multiplied.
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    img_pad = cv2.copyMakeBorder(img_f, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    otf = psf_to_otf(kernel, img_pad.shape)

    F = np.fft.fft2(img_pad)
    Hf = np.fft.fft2(otf)
    G = F * Hf

    def log_mag(X):
        return np.log1p(np.abs(np.fft.fftshift(X)))

    # ---- Plot ----
    plt.figure(figsize=(14, 9))

    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(img_f, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Spatial convolution result")
    plt.imshow(spatial, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Frequency multiplication result")
    plt.imshow(freq, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("log|FFT(image)| (display)")
    plt.imshow(log_mag(F), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("log|FFT(kernel)| (display)")
    plt.imshow(log_mag(Hf), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("|Spatial - Frequency|")
    plt.imshow(diff, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
