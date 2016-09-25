from palette_extractor import palette_extractor as pe
import numpy as np

def t_image():
    test_img = np.zeros((400,400,3), np.uint8)
    test_img[0:100,0:] = (0, 255, 0)
    test_img[100:200,0:] = (255, 0, 0)
    test_img[200:300,0:] = (0, 0, 255)
    test_img[300:,0:] = (0, 255, 255)
    return test_img

def test_image_segmentation():
    color, ret = pe.image_segmentation(k = 4, image=t_image())
    assert color == [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
    assert ret.shape == (400, 400, 3)


def test_palette():
    color, ret = pe.image_segmentation(k = 4, image=t_image())
    ret = pe.palette(color, ret, x_len=10)
    assert ret.shape == (400, 40, 3)

def test_concatenate_images():
    color, img = pe.image_segmentation(k = 4, image=t_image())
    img2 = pe.palette(color, img, x_len=10)
    vis = pe.concatenate_images(img, img2)
    assert vis.shape == (400, 440, 3)