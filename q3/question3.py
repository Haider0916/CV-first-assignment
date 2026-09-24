from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = os.path.join(os.path.dirname(__file__), "sample_img.png")


def preprocessImage(img):
    # same thing as q1, grayscale then divide by 255
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    arr = arr / 255.0
    return arr


def downSampleImage(normalizedImgArr):
    downsampledImageArr = []
    for i in range(len(normalizedImgArr)):
        if i % 2 == 0:
            downsampledImageArr.append([])  # Initialize an empty list for each row
            for j in range(len(normalizedImgArr[i])):
                if j % 2 == 0:
                    downsampledImageArr[-1].append(normalizedImgArr[i][j])

    downsampledImageNP = np.array(downsampledImageArr)  # list -> array
    return downsampledImageNP  # keep it as an array in [0,1] for further processing


# showing the images together side by side
def showImage(normalizedImgArr, SampleImage1, SampleImage2, operation, filter):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(
        axes,
        [normalizedImgArr, SampleImage1, SampleImage2],
        ["Original", f"{operation}d + {filter} once", f"{operation}d + {filter} twice"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title} {im.shape}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"q3_{operation}_{filter}.png", dpi=150)
    plt.show()


def upSampleImage(normalizedImgArr):

    h, w = normalizedImgArr.shape
    upsampledImageArr = np.zeros((h * 2, w * 2))

    for i in range(h):
        for j in range(w):
            upsampledImageArr[i * 2, j * 2] = normalizedImgArr[
                i, j
            ]  # copy the pixel value to the top-left corner of the 2x2 block
    return upsampledImageArr


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


img = Image.open(image_path)

normalizedImgArr = preprocessImage(img)

downSampleImage1 = downSampleImage(normalizedImgArr)  # first downsample
downSampleImage2 = downSampleImage(downSampleImage1)  # second downsample

upSampleImage1 = upSampleImage(downSampleImage2)  # first upsample


# gaussian smoothing after upsampling - start
gaussianSmoothedUpSampledImage1 = gaussianSmoothing(
    upSampleImage1, 11, 1
)  # first smoothing using gaussian smoothing

upSampleImage2 = upSampleImage(gaussianSmoothedUpSampledImage1)  # second upsample

gaussianSmoothedUpSampledImage2 = gaussianSmoothing(
    upSampleImage2, 11, 1
)  # second smoothing using gaussian smoothing

showImage(
    normalizedImgArr,
    gaussianSmoothedUpSampledImage1,
    gaussianSmoothedUpSampledImage2,
    operation="upsample",
    filter="gaussian",
)
# gaussian smoothing after upsampling - end


# median smoothing after upsampling - start
medianSmoothedUpSampledImage1 = medianSmoothing(
    upSampleImage1
)  # first smoothing using median smoothing

upSampleImage2 = upSampleImage(medianSmoothedUpSampledImage1)  # second upsample

medianSmoothedUpSampledImage2 = medianSmoothing(
    upSampleImage2
)  # second smoothing using median smoothing

showImage(
    normalizedImgArr,
    medianSmoothedUpSampledImage1,
    medianSmoothedUpSampledImage2,
    operation="upsample",
    filter="median",
)

# median smoothing after upsampling - end
