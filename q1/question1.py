from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = os.path.join(os.path.dirname(__file__), "sample_img.png")


# showing the images together side by side for better comparison
def showImage(normalizedImgArr, SampleImage1, SampleImage2, operation):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(
        axes,
        [normalizedImgArr, SampleImage1, SampleImage2],
        ["Original", f"{operation}d once", f"{operation}d twice"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title} {im.shape}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"q1_{operation}.png", dpi=150)
    plt.show()


def preprocessImage(img):
    gray_img = img.convert("L")  # convert image to grayscale

    ImgArr = np.array(
        gray_img, dtype=np.float32
    )  # convert image to numpy array and set data type to float32

    normalizedImgArr = ImgArr / 255.0  # convert pixel values to [0,1] range

    return normalizedImgArr


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


def upSampleImage(normalizedImgArr):

    h, w = normalizedImgArr.shape
    upsampledImageArr = np.zeros((h * 2, w * 2))

    for i in range(h):
        for j in range(w):
            upsampledImageArr[i * 2, j * 2] = normalizedImgArr[
                i, j
            ]  # copy the pixel value to the top-left corner of the 2x2 block
    return upsampledImageArr


img = Image.open(image_path)

normalizedImgArr = preprocessImage(img)

downSampleImage1 = downSampleImage(normalizedImgArr)  # first downsample
downSampleImage2 = downSampleImage(downSampleImage1)  # second downsample

showImage(normalizedImgArr, downSampleImage1, downSampleImage2, operation="downsample")

upSampleImage1 = upSampleImage(downSampleImage2)  # first upsample
upSampleImage2 = upSampleImage(upSampleImage1)  # second upsample

showImage(normalizedImgArr, upSampleImage1, upSampleImage2, operation="upsample")
