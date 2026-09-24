from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

image_path = os.path.join(os.path.dirname(__file__), "sample_img.png")
Gx = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float
)  # right side minus left side
Gy = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float
)  # bottom row minus top row


def preprocessImage(img):
    gray_img = img.convert("L")  # convert image to grayscale

    ImgArr = np.array(
        gray_img, dtype=np.float32
    )  # convert image to numpy array and set data type to float32

    normalizedImgArr = ImgArr / 255.0  # convert pixel values to [0,1] range

    return normalizedImgArr


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


def applyKernel(I, kernel):
    k = kernel.shape[0]
    p = k // 2
    padded = padImage(I, p)

    h, w = I.shape
    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            window = padded[i : i + k, j : j + k]
            output[i][j] = np.sum(window * kernel)

    return output


def mySobelFilter(I):
    gx = applyKernel(I, Gx)  # change going left to right, at every pixel
    gy = applyKernel(I, Gy)  # change going top to bottom, at every pixel

    mag = np.sqrt(gx**2 + gy**2)  # edge strength = length of the (gx, gy) arrow
    ori = np.arctan2(gy, gx)  # edge direction = angle of the arrow, from -pi to pi

    mag = mag / mag.max()  # 0 = no edge, 1 = strongest edge
    ori = (ori + np.pi) / (2 * np.pi)  # -pi..pi  ->  0..1

    return mag, ori


img = Image.open(image_path)
I = preprocessImage(img)
mag, ori = mySobelFilter(I)


gx = applyKernel(I, Gx)
gy = applyKernel(I, Gy)
gxShow = (gx - gx.min()) / (
    gx.max() - gx.min()
)  # squeeze into [0,1]: 0.5 gray = no change
gyShow = (gy - gy.min()) / (gy.max() - gy.min())


hsv = np.stack(
    [ori, mag, mag], axis=2
)  # layer 1 = H (direction), 2 = S (strength), 3 = V (strength)
colorImg = hsv_to_rgb(hsv)  # convert to normal screen colors


showImages(
    [I, gxShow, gyShow, mag, ori, colorImg],
    ["Original", "Gradient x", "Gradient y", "Magnitude", "Orientation", "HSV color"],
    "q5_sobel.png",
)
