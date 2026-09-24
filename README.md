# CSCI-B 657 Computer Vision — Project 1

**Author:** Syed Haider Imam Abidi

## Requirements

- Python 3
- numpy, pillow, matplotlib

```
pip install numpy pillow matplotlib
```

## Folder structure

```
q1/   question1.py, sample_img.png
q2/   question2.py, sample_img.png
q3/   question3.py, sample_img.png
q4/   (to be added)
q5/   (to be added)
report.pdf
```

## Problem 1: Sampling

```
cd q1
python question1.py
```

- Loads `sample_img.png` (228×228 Lena) from the same folder.
- To test another image, replace `sample_img.png` or change the filename
  in the `image_path` line near the top of `question1.py`.
- Two figure windows open one after the other (close the first to see the second).
- Saves `q1_downsample.png` and `q1_upsample.png` in the current folder.

## Problem 2: Gaussian Smoothing

```
cd q2
python question2.py
```

- Loads `sample_img.png` from the same folder (replace it or edit the
  `image_path` line to test another image).
- Main function: `myGaussianSmoothing(I, k, s)`, where `I` is a grayscale
  image in [0, 1], `k` is the kernel size (odd) and `s` is sigma.
- Runs 10 experiments on the original image:
  - k = 3, 5, 7, 11, 51 with s = 1
  - s = 0.1, 1, 2, 3, 5 with k = 11
- Prints progress ("doing k = ...") while it runs; all 10 runs take a few
  seconds on a 228×228 image (k = 51 is the slowest).
- Two figure windows open one after the other (close the first to see the second).
- Saves `q2_vary_k.png` and `q2_vary_sigma.png` in the current folder.

## Problem 3: Image Filtering

```
cd q3
python question3.py
```

- Loads `sample_img.png` from the same folder, downsamples it twice (as in
  Problem 1), then upsamples it twice with a filter after every upsampling step.
- Gaussian version: `gaussianSmoothing` with k = 11, sigma = 1.
- Median version: my own `medianSmoothing` with a 3×3 window (k = 3).
- Two figure windows open one after the other (close the first to see the second).
- Saves `q3_upsample_gaussian.png` and `q3_upsample_median.png` in the current folder.
- Runs in a couple of seconds.

## Problems 4–5

To be added.