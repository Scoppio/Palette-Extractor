# Palette-Extractor
This program was created for my wife, it extracts an N colors palette of any image sourced to it.

## How it was made and how to install?
This program features Pillow :D

First you have to install the requirements, that is basically Pillow.

## Installation procedure
* pip install -r requirements.txt or pip install Pillow
* pip install setup.py

## How to use
You can use the script as a standalone and just feed it with its arguments, or as a package.

### Arguments

---

* --image / -i :: image sourced

* --colors_palette / -c :: number of colors to extract

* --silent / -s :: silent mode, do not show preview window

---


Palette Extractor
--------

To use do::

    >>> from palette_extractor import *
	>>> app = PaletteSampler()
	>>> app.load_image(image_file_path)
    >>> number_of_colors = 3
	>>> image, color_255, color_hexa = app.render(n=number_of_colors)
