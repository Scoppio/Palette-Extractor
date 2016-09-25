# Palette-Extractor
This program was created for my wife, it extracts an N colors palette of any image sourced to it.

## How it was made and how to install?
This program features OpenCV, altought I think I could do it with Pillow, I have more experience with CV2,
but due to its easy install I guess I should use Pillow more times.

First you have to install the requirements, that are basically numpy and cv2. I sugest that you download a precompiled version of both,
either in the sources for ubuntu or precompiled version for windows/mac... really.

## Installation procedure
* pip install -r requirements.txt
* pip install setup.py

## How to use
You can use the script as a standalone and just feed it with its arguments, or as a package.

### Arguments

---
image: -i image sourced

colors_palette: -c number of colors to extract

silent: -s silent mode, do not show preview window
---


Palette Extractor
--------

To use do::

    >>> from palette_extractor import *
    >>> img = cv2.imread("filename with its path")
    >>> colour, res2 = image_segmentation(k = args.colors_palette, image=img)
    >>> img2 = palette(colour, res2)
    >>> vis = concatenate_images(img, img2)
    >>> cv2.imshow('output',vis)
