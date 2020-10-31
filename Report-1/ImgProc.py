import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from PIL import Image


# constants
IMIN = np.uint8(0) # black
IMAX = np.uint8(255) # white

class MyImgProcLib:
    def __init__(self, show=True, verbose=True):
        self.show = show
        self.verbose = verbose

    def load(self, filepath):
        # load an image
        img = Image.open(filepath)
        # get info
        width, height = img.size
        mode = 'B&W' if img.mode == 'L' else img.mode
        # convert into a numpy array
        p = np.array(img, dtype=np.uint8)

        if self.show: img.show()
        if self.verbose: print(f'Image loaded: {filepath}, ({width} x {height}), ({mode})')

        return p

    def save(self, p, filepath):
        pass

    def imshow(self, p):
        img = Image.fromarray(p)
        img.show()

    def add_salt_and_pepper(self, p, ratio_salt=0.05, ratio_pepper=0.05):
        # noised image data
        pn = p.copy()
        height, width = pn.shape

        # calculate no. of pixels to be noised
        n_salt = np.int(width * height * ratio_salt)
        n_pepper = np.int(width * height * ratio_pepper)
        # generate noised image data
        for _ in range(n_salt):
            x, y = rd.randint(IMAX + 1), rd.randint(IMAX + 1)
            pn[x,y] = IMAX
        for _ in range(n_pepper):
            x, y = rd.randint(IMAX + 1), rd.randint(IMAX + 1)
            pn[x,y] = IMIN

        if self.show: self.imshow(pn)
        if self.verbose: print(f'Added salt-and-pepper noise: no. of salt, pepper = ({n_salt}, {n_pepper})')

        return pn

    def median_filter(self, p, window_size=3):
        # denoise by median filter
        po = np.zeros_like(p)
        height, width = p.shape
        ofs = window_size // 2 # (window size) / 2 - 0.5

        for y in range(height):
            for x in range(width):
                # window coordinates
                left = max(x - ofs, 0)
                right = min(x + ofs, width - 1)
                top = max(y - ofs, 0)
                bottom = min(y + ofs, height - 1)
                # get window pixel data and flatten
                p_win = p[left:right, top:bottom].flatten()
                # calculate index to median pixel
                imed = len(p_win) // 2
                # sort and extract median pixel data
                p_med = sorted(p_win)[imed]
                # assign to the output image array
                po[x,y] = p_med
        
        if self.show: self.imshow(po)
        if self.verbose: print(f'Image is denoise with a median filter (window size = {window_size})')

        return po
