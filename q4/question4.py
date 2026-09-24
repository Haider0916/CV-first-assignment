from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = os.path.join(os.path.dirname(__file__), "sample_img.png")


def preprocessImage(img):
    gray_img = img.convert("L")  # convert image to grayscale

    ImgArr = np.array(
        gray_img, dtype=np.float32
    )  # convert image to numpy array and set data type to float32

    normalizedImgArr = ImgArr / 255.0  # convert pixel values to [0,1] range

    return normalizedImgArr


def gaussianKernel(k, s):
    kernel = np.zeros((k, k))
    c = k // 2  # middle index

    # to get the kernel values
    for i in range(k):
        for j in range(k):
            x = j - c  # horizontal distance from the center
            y = i - c  # vertical distance from the center
            kernel[i][j] = np.exp(-(x**2 + y**2) / (2 * s**2))

    # normalize the kernel so that everything adds to 1
    total = 0
    for i in range(k):
        for j in range(k):
            total += kernel[i][j]
    kernel = kernel / total

    return kernel


def padImage(normalizedImgArr, p):
    h, w = normalizedImgArr.shape
    padded = np.zeros((h + 2 * p, w + 2 * p))

    for i in range(h + 2 * p):
        for j in range(w + 2 * p):
            # position in the original image, clamped to stay inside it
            srcRow = min(max(i - p, 0), h - 1)
            srcCol = min(max(j - p, 0), w - 1)
            padded[i][j] = normalizedImgArr[srcRow][srcCol]

    return padded


def gaussianSmoothing(I, k, s):
    kernel = gaussianKernel(k, s)
    p = k // 2
    padded = padImage(I, p)

    h, w = I.shape
    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            window = padded[i : i + k, j : j + k]
            output[i][j] = np.sum(window * kernel)

    return output


def medianSmoothing(I, k=3):
    p = k // 2
    padded = padImage(I, p)

    h, w = I.shape
    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            window = padded[i : i + k, j : j + k]
            output[i][j] = (np.sort(window.flatten()))[k * k // 2]  # median value

    return output


def showImages(images, titles, fileName):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    idx = 0
    for r in range(2):
        for c in range(3):
            axes[r][c].imshow(
                images[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
            axes[r][c].set_title(titles[idx])
            axes[r][c].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(fileName, dpi=150)
    plt.show()


img = Image.open(image_path)
I = preprocessImage(img)

np.random.seed(0)  # seeding the random number generator for reproducibility
r = np.random.normal(
    0, 0.1, I.shape
)  # gaussian noise - make one random number per pixel: mean 0, std dev 0.1 (the question's r ~ N(0, 0.1))

noisy1 = np.clip(I + r, 0, 1)  # adds Gaussian noise to the original image

noisy1Gaussian = gaussianSmoothing(noisy1, 7, 1)

noisy1Median = medianSmoothing(noisy1, 3)

salt = (r > 0.2).astype(
    float
)  # salt noise - add 1 everywhere where the noise was > 0.2 and 0 otherwise

noisy2 = np.clip(I + salt, 0, 1)  # adds salt noise to the original image

noisy2Gaussian = gaussianSmoothing(noisy2, 7, 1)
noisy2Median = medianSmoothing(noisy2, 3)


showImages(
    [noisy1, noisy1Gaussian, noisy1Median, noisy2, noisy2Gaussian, noisy2Median],
    [
        "Gaussian noise",
        "Gaussian noise + Gaussian",
        "Gaussian noise + median",
        "Salt noise",
        "Salt noise + Gaussian",
        "Salt noise + median",
    ],
    "q4_noise.png",
)
