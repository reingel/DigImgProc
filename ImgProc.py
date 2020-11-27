import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from PIL import Image


# constants
nBit = np.uint8(8) # number of bits to represent a pixel
L = np.uint(2**nBit) # number of gray-scale levels
IMIN = np.uint8(0) # black
IMAX = np.uint8(L - 1) # white

class ImgProcLib:
    def __init__(self, show=False, verbose=True):
        self.show = show
        self.verbose = verbose

    def load(self, filepath):
        # load an image
        img = Image.open(filepath)
        # get info
        width, height = img.size
        mode = 'B&W' if img.mode == 'L' else img.mode
        # convert into a numpy array
        f = np.array(img, dtype=np.uint8)

        if self.show: img.show(title='original image')
        if self.verbose: print(f'Image loaded: {filepath}, ({width} x {height}), ({mode})')

        return f

    def save(self, f, filepath):
        pass

    def imshow(self, f, title=None):
        img = Image.fromarray(f)
        img.show(title=title)
    
    def get_pixel_abs_win(self, f, left, right, top, bottom, flatten=False):
        height, width = f.shape

        x_min, x_max = max(left, 0), min(right, width - 1)
        y_min, y_max = max(top, 0), min(bottom, height - 1)

        f_win = f[y_min:y_max+1, x_min:x_max+1] # NOTE: f[y index, x index]
        if flatten: f_win = f_win.flatten()
        is_boundary = (x_max - x_min < right - left) or (y_max - y_min < bottom - top)

        return f_win.copy(), is_boundary

    def get_pixel_rel_win(self, f, y, x, ofs, flatten=False):
        return self.get_pixel_abs_win(f, x - ofs, x + ofs, y - ofs, y + ofs, flatten)
    
    def inverse(self, f):
        # inverse image
        g = f.copy()
        height, width = f.shape

        for y in range(height):
            for x in range(width):
                g[y, x] = IMAX - f[y, x]
        
        return g
    
    def difference(self, f1, f2, neg_proc='abs'):
        # generate difference image between two images
        diff = np.array(f1, dtype=np.int) - np.array(f2, dtype=np.int)

        if neg_proc == 'abs':
            g = np.array(np.abs(diff), dtype=np.uint8)
        elif neg_proc == 'clip':
            g = np.array(np.maximum(diff, 0), dtype=np.uint8)
        else:
            g = None

        return g

    def histogram(self, f):
        # return histogram of input image
        g = f.flatten()
        N = len(g)

        # array for histogram
        h = np.zeros(L, dtype=np.uint16)
        # accumulate for every pixel
        for s in range(N):
            level = g[s]
            h[level] += 1
        
        return h
    
    def prob_dist(self, f):
        N = len(f.flatten())
        h = self.histogram(f)
        h = np.array(h)
        return h / N
    
    def threshold_otsu(self, f):
        p = self.prob_dist(f)
        w = np.cumsum(p) # class probability vector
        w[(w == 0) + (w == 1)] = 1e-9
        u = np.cumsum(range(L)*p) # class mean vector
        uT = u[-1] # total mean
        sB2 = (uT*w - u)**2 / (w * (1 - w)) # between-class variance vector
        k_star = np.argmax(sB2)
        g = np.zeros_like(f)
        g[f > k_star] = IMAX
        return g, p, sB2, k_star