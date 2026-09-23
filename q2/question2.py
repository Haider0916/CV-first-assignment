from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# image is in the same folder as this file
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "sample_img.png")


def preprocessImage(img):
    # same thing as q1, grayscale then divide by 255
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    arr = arr / 255.0
    return arr


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


# This image pads the image with zeros around the edges, so that when we apply the kernel, we dont go out of bounds of the image. The padding is done by replication the edge pixels of the image. The amount of padding is determined by the size of the kernel. For example, if the kernel is 3x3, we need to pad the image with 1 pixel on each side. If the kernel is 5x5, we need to pad the image with 2 pixels on each side, and so on.
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


def myGaussianSmoothing(I, k, s):
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

# part 1 of the assignment, changing k, constant s at 1
images1 = [I]
titles1 = ["Original"]
for k in [3, 5, 7, 11, 51]:
    print("doing k =", k)
    result = myGaussianSmoothing(I, k, 1)
    images1.append(result)
    titles1.append("k = " + str(k) + ", s = 1")
showImages(images1, titles1, "q2_vary_k.png")

# part 2 of the assignment - changing sigma, k stays constant at 11
images2 = [I]
titles2 = ["Original"]
for s in [0.1, 1, 2, 3, 5]:
    print("doing s =", s)
    result = myGaussianSmoothing(I, 11, s)
    images2.append(result)
    titles2.append(f"k = 11, s = {s}")
showImages(images2, titles2, "q2_vary_sigma.png")
