import numpy as np
import cv2
import colorsys
import argparse
import math
import os


def image_segmentation(k, image):
  """image_segmentation(k=int, image=nparray)
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

  Z = image.reshape((-1,3))
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  # k defines the number of main colors to capture in the image
  _ , label ,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  res = center[label.flatten()]
  res2 = res.reshape((image.shape))
  colour = []

  for c in center:
    colour.append((int(c[0]),int(c[1]),int(c[2]) ))

  colour.sort(key= lambda x: step(x[0],x[1],x[2],8))

  return colour, res2

def palette(colour, image, x_len = 6):
  """palette(colour, image)
  colour: array of tuples with color information
  image: is the source image (cv2) to capture its x and y size
  return -> generated palette image"""
  
  y,x,_ = image.shape
  img2 = np.zeros((int(y), int(x/x_len),3), np.uint8)
  
  x = int(x/x_len)
  
  for centroid_color, pos in zip(colour, range(0,len(colour))):
    cv2.rectangle(img2, (0, int(y-pos*y/len(colour) - y/len(colour))), (x, int(y-pos*y/len(colour) - y/len(colour)) + int(y/len(colour))), 
      color=(centroid_color), thickness=-1)  
 
  return img2

def concatenate_images(img, img2):
  """enter two images and it concatenates side-by-side
  return -> concatenated images"""
  
  h1, w1 = img.shape[:2]
  h2, w2 = img2.shape[:2]
  vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
  vis[:h1, :w1] = img
  vis[:h2, w1:w1+w2] = img2

  return vis

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process the a reduced color palette of an image.')
  parser.add_argument('-i','--image',type=str, required=True,
                    help='the file to extract the palette')
  parser.add_argument('-c','--colors_palette', type=int, default=5,
                    help='number of colors you want to extract')
  parser.add_argument('-s', '--silent', action='store_true', default=False)
  args = parser.parse_args()

  
  img = cv2.imread(args.image)
  
  colour, res2 = image_segmentation(k = args.colors_palette, image=img)

  img2 = palette(colour, res2)
  
  vis = concatenate_images(img, img2)

  file_path, file_ext = os.path.splitext(args.image)
  output_file = file_path+'_palette='+str(args.colors_palette)+file_ext
  cv2.imwrite(output_file, vis)

  if not args.silent:
    cv2.imshow('res2',vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
