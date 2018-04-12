import math
import random
import colorsys
import sys
import os
from math import sqrt
from collections import namedtuple
from PIL import Image, ImageDraw

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PaletteSampler():
    __metaclass__ = Singleton
    im_filename_ = None
    im_ = None
    palette_ = None
    new_im = None

    def get_points(self):
        points = []
        w, h = self.im_.size
        for count, color in self.im_.getcolors(w * h):
            points.append(Point(color, 3, count))
        return points

    def colorz(self, n=5):
        rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

        def step(r, g, b, repetitions=1):
            """step color sorting"""
            lum = math.sqrt(.241 * r + .691 * g + .068 * b)

            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            h2 = int(h * repetitions)
            lum2 = int(lum * repetitions)
            v2 = int(v * repetitions)

            return (h2, lum, v2)

        img = self.im_.copy()
        img.thumbnail((200, 200))
        w, h = img.size

        points = self.get_points()
        clusters = self.kmeans(points, n, 1)

        rgbs = [c.center.coords for c in clusters]

        rgbs.sort(key=lambda x: step(int(x[0]), int(x[1]), int(x[2]), 8))
        hexa = map(rtoh, rgbs)

        return hexa, rgbs

    def euclidean(self, p1, p2):
        return sqrt(sum([
            (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
        ]))

    def calculate_center(self, points, n):
        vals = [0.0 for i in range(n)]
        plen = 0
        for p in points:
            plen += p.ct
            for i in range(n):
                vals[i] += (p.coords[i] * p.ct)
        return Point([(v / plen) for v in vals], n, 1)

    def kmeans(self, points, k, min_diff):
        clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]
        idx = 0
        while 1:
            plists = [[] for i in range(k)]

            for p in points:
                smallest_distance = float('Inf')
                for i in range(k):
                    distance = self.euclidean(p, clusters[i].center)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        idx = i
                plists[idx].append(p)

            diff = 0
            for i in range(k):
                old = clusters[i]
                center = self.calculate_center(plists[i], old.n)
                new = Cluster(plists[i], center, old.n)
                clusters[i] = new
                diff = max(diff, self.euclidean(old.center, new.center))

            if diff < min_diff:
                break

        return clusters

    def load_image(self, filename):
        self.im_filename_ = filename
        self.im_ = Image.open(filename)

    def save_image(self, filename):
        if self.new_im is not None:
            self.new_im.save(filename)
        else:
            print("Error, no file loaded in memory")

    def render(self, n=3, percent=0.1, output="~temp.png"):
        self.im_ = Image.open(self.im_filename_)
        w, h = self.im_.size
        b, a = self.colorz(n=n)
        palette_w = int(w * percent)
        palette_y = int(h / n)
        size = (palette_w, h)
        self.palette_ = Image.new('RGB', size)

        xy = [0, 0, palette_w, palette_y]
        draw = ImageDraw.Draw(self.palette_)

        draw.rectangle([0, 0, int(palette_w), int(h)], fill=tuple([int(x) for x in a[-1]]), outline=None)

        for colour in a:
            draw.rectangle(xy, fill=tuple([int(x) for x in colour]), outline=None)
            xy[1] += palette_y
            xy[3] += palette_y

        del draw

        images = [self.im_, self.palette_]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        self.new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            self.new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        self.new_im.save(output)

        return self.new_im, a, b


if __name__ == '__main__':
    image_path = sys.argv[1]
    n = int(sys.argv[2])
    app = PaletteSampler()
    app.load_image(image_path)
    output_path = os.path.join(os.path.split(image_path)[0], "sampled_" + str(n) + "_" + os.path.split(image_path)[1])
    app.render(n=n, output=output_path)

