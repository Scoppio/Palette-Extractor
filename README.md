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

Arguments
---
-i/--image: image sourced
-c/--colors_palette: number of colors to extract
-s/--silent: silent mode, do not show preview window
---

Check the test file and the script on how to use it.
