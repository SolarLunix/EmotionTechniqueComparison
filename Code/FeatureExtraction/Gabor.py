import math
import numpy as np
import cv2 as cv

name = "Gabor"


class featureExtraction():

    def __init__(self, params):
        self.n_filters = params['n_filters']
        k = int(params['k_size']/2)
        self.x_max = k
        self.x_min = -k
        self.y_max = k
        self.y_min = -k
        self.lamb = params['lamb']
        self.psi = params['psi']
        self.sigma = params['sigma']
        self.gamma = params['gamma']

        self.filters = []
        self.create_filters()

    def runAll(self, images):
        for i, img in enumerate(images):
            images[i] = np.array(self.gabor(images[i])).flatten()
        return images

    def gabor(self, img):
        new_img = np.zeros_like(img)
        for filter1 in self.filters:
            f_img = cv.filter2D(img, cv.CV_8UC3, filter1)
            np.maximum(new_img, f_img, new_img)
        return new_img

    def x_mark(self, x, y, t):
        ct = math.cos(t)
        st = math.sin(t)
        xm = (x * ct) + (y * st)
        return xm

    def y_mark(self, x, y, t):
        ct = math.cos(t)
        st = math.sin(t)
        ym = (-x * st) + (y * ct)
        return ym

    def pixel_value(self, x, y, t):
        xm = self.x_mark(x, y, t)
        ym = self.y_mark(x, y, t)
        lhu = -(xm ** 2 + (self.gamma ** 2 * ym ** 2))
        lhl = 2 * self.sigma ** 2
        lh = math.exp(lhu / lhl)
        rhi = (2 * math.pi * (xm / self.lamb)) + self.psi
        rh = math.cos(rhi)
        p = lh * rh
        return p

    def create_filters(self):
        d = 180 / self.n_filters
        for i in range(0, self.n_filters):
            t = i * d
            filt = []
            for j in range(self.x_min, self.x_max):
                row = []
                for c in range(self.y_min, self.y_max):
                    pixel = self.pixel_value(c, j, t)
                    row.append(pixel)
                filt.append(row)
            n_filter = np.array(filt)
            self.filters.append(n_filter)
 