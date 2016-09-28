import numpy as np
import cv2
import colorsys
import argparse
import math
import os

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PaletteException(Exception):
    pass

class PaletteExtractor(object):
    __metaclass__ = Singleton

    def __init__(self, image = None, k=5, screensize=(600,800), palette_position = "right", palette_width = 0.16):
        self._image = image
        self._screensize = {}
        self._screensize["height"] = screensize[1] # y
        self._screensize["width"] = screensize[0]  # x
        self._color_palette = None
        self._image_palette = None

        if palette_width >0.1 and palette_width < 0.5:
            self._palette_width = palette_width
        else:
            raise PaletteException({"message": "palette_width must be between 0.1 and 0.5, default 0.16 designated", "value":palette_width})
            self._palette_width = 0.16

        if k > 1 and k < 256:
            self._k = k
        else:
            self._k = 5

        if image is not None:
            self._img_orientation = "portrait" if image.shape[0] > image.shape[1] else "landscape"
        else:
            self._img_orientation = None

        if palette_position is not "right" or palette_position is not "left":
            raise PaletteException({"message":"palette_position only allows for 'left' or 'right', default 'right' is designated",
                                    "argument":palette_position})
            self._palette_position = "right"
        else:
            self._palette_position = palette_position

    def _imageSegmentation(self):
        """"image_segmentation(k=int, image=nparray)
        k: is the number of colors
        image: is the source image (cv2)
        return -> color array, image segmentated"""

        def step (r,g,b, repetitions=1):
            """step color sorting"""
            lum = math.sqrt( .241 * r + .691 * g + .068 * b )

            h, s, v = colorsys.rgb_to_hsv(r,g,b)

            h2 = int(h * repetitions)
            lum2 = int(lum * repetitions)
            v2 = int(v * repetitions)

            return (h2, lum, v2)

        k = self._k
        Z = self._image.reshape((-1,3))
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # k defines the number of main colors to capture in the image
        _ , _ ,center = cv2.kmeans(Z,k,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # res = center[label.flatten()]
        # imseg = res.reshape((image.shape))
        colour = []

        for c in center:
            colour.append((int(c[0]),int(c[1]),int(c[2]) ))

        colour.sort(key= lambda x: step(x[0],x[1],x[2],8))

        self._color_palette = colour

        return colour # , imseg

    def _paletteCreation(self):
        """palette(self)
        colour: array of tuples with color information
        image: is the source image (cv2) to capture its x and y size
        return -> generated palette image"""

        colour = self._color_palette

        y,x = self._image.shape[0], int(self._image.shape[1] * self._palette_width)
        img = np.zeros((int(y), int(x), 3), np.uint8)

        for centroid_color, pos in zip(colour, range(0,len(colour))):
            cv2.rectangle(img, (0, int(y-pos*y/len(colour) - y/len(colour))),
                (x, int(y-pos*y/len(colour) - y/len(colour)) + int(y/len(colour))),
                color=(centroid_color), thickness=-1)

        self._image_palette = img

        return img

    def changePaletteWidth(self, width):
        if width > 0.5 or int(width*self._image.shape[1]) < 35:
            raise PaletteException({"message":"width too big or too small","width":width})
        else:
            self._palette_width = width

    def changePalettePosition(self, pos):
        if pos == "right" or pos == "left":
            self._palette_position = pos
        else:
            raise PaletteException({"message":"Only accepts 'right' or 'left' as argument","argument":pos})

    def changeK(self, k):
        if type(k) == int:
            if k < 1 or k > 255 or k > self._image.shape[0]:
                raise PaletteException({"message":"k value must be a POSITIVE integer SMALLER THAN 256",
                                        "size":k})
            else:
                self._k = k
        else:
            raise PaletteException({"message":"k value must be a positive INTEGER smaller than 256","type":type(k)})

    def loadImage(self, image):
        if image is not None:
            __init__(image=image, k=self._k, screensize=self._screensize,
                        palette_position=self._palette_position,
                        palette_width=self._palette_width)
        else:
            raise PaletteException({"message":"There was an error with the image", "error":"Image not present"})

    def renderConcatImage(self):
        """enter two images and it concatenates side-by-side
        return -> concatenated images"""

        vis = None

        if self._image is None or self._image_palette is None:
            if self._image is None:
                raise PaletteException({"message":"There is no image loaded"})

            if self._image_palette is None:
                raise PaletteException({"message":"Palette was not rendered"})
        else:
            img = self._image.copy
            pal = self._image_palette.copy

            h1, w1 = self._image.shape[:2] if self._palette_position == "right" else self._image_palette.shape[:2]
            h2, w2 = self._image_palette.shape[:2] if self._palette_position == "right" else self._image.shape[:2]

            vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
            vis[:h1, :w1] = self._image if self._palette_position == "right" else self._image_palette
            vis[:h2, w1:w1+w2] = self._image_palette if self._palette_position == "right" else self._image

            if self._img_orientation == "portrait":
                if vis.shape[0] > self._screensize["height"]:
                    vis = resize(vis, fixedHeight=self._screensize["height"])
                if vis.shape[1] > self._screensize["width"]:
                    vis = resize(vis, fixedWidth=self._screensize["width"])
            else:
                if vis.shape[1] > self._screensize["width"]:
                    vis = resize(vis, fixedWidth=self._screensize["width"])
                if vis.shape[0] > self._screensize["height"]:
                    vis = resize(vis, fixedHeight=self._screensize["height"])

        return vis

    def process(self):
        result = False
        if self._image is not None:
            self._imageSegmentation()
            self._paletteCreation()
            if self._image_palette is not None:
                result = True
        else:
            raise PaletteException({"message":"Image is not loaded"})

        return result

def resize(frame, scale=None, fx=0, fy=0, fixedWidth=None, fixedHeight=None, interpolation=cv2.INTER_CUBIC):
  '''
  Resize image
  size          = Resize image with fixed dimension
  scale         = If set, ignores the value of `size` and resize dividing the image dimensions
  fx            = scale factor along the horizontal axis
  fy            = scale factor along the vertical axis
  fixedWidth    = resize image by width keeping aspect ratio
  fixedHeight   = resize image by height keeping aspect ratio
  interpolation = default is bicubic interpolation over 4x4 pixel neighborhood
  '''

  if fixedWidth:
	imh, imw = frame.shape[:2]
	new_height = int(float(fixedWidth) * (float(imh) / float(imw)))
	sz = (fixedWidth, new_height)

  if fixedHeight:
 	imh, imw = frame.shape[:2]
	new_width  = int(float(fixedHeight) * (float(imw) / float(imh)))
	sz = (new_width, fixedHeight)

  if scale:
	sz = (int(frame.shape[1] // scale), int(frame.shape[0] // scale))

  scaled_image = cv2.resize(frame, sz, fx=fx, fy=fy, interpolation=interpolation)

  return scaled_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the a reduced color palette of an image.')
    parser.add_argument('-i','--image',type=str, required=True,
                    help='the file to extract the palette')
    parser.add_argument('-c','--colors_palette', type=int, default=5,
                    help='number of colors you want to extract')
    parser.add_argument('-s', '--silent', action='store_true', default=False)
    args = parser.parse_args()

    img = cv2.imread(args.image) # Loads the image

    # Creates the PaletteExtractor object ans initializes it with mandatory initialization
    app_palette = PaletteExtractor(image=img, k=5, screensize=(800,600), palette_position="Left")

    # Run image process
    app_palette.process()

    # Render the image
    vis = app_palette.renderConcatImage()

    file_path, file_ext = os.path.splitext(args.image)
    output_file = file_path+'_palette='+str(args.colors_palette)+file_ext
    cv2.imwrite(output_file, vis)

    if not args.silent:
      cv2.imshow(str(vis.shape),vis)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
