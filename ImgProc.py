import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from PIL import Image


# constants
IMIN = np.uint8(0) # black
IMAX = np.uint8(255) # white

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

    def add_salt_and_pepper(self, f, ratio_salt=0.05, ratio_pepper=0.05):
        # add salt-and-pepper noise
        fn = f.copy()
        height, width = fn.shape

        # calculate no. of pixels to be noised
        n_salt = np.int(width * height * ratio_salt)
        n_pepper = np.int(width * height * ratio_pepper)
        # add salt noises
        for _ in range(n_salt):
            y, x = rd.randint(height), rd.randint(width)
            fn[y, x] = IMAX
        # add pepper noises
        for _ in range(n_pepper):
            y, x = rd.randint(height), rd.randint(width)
            fn[y, x] = IMIN

        if self.show: self.imshow(fn, title='Image with salt-and-pepper noise')
        if self.verbose: print(f'Added salt-and-pepper noise: no. of salt, pepper = ({n_salt}, {n_pepper})')

        return fn

    def median_filter(self, f, win_size=3, inc_bound=True):
        #
        # median filter
        #
        # f: input image data
        # win_size: filter size
        # inc_bound: filter boundary if True
        # g: filtered image data
        #
        g = np.zeros_like(f)
        height, width = f.shape
        ofs = win_size // 2 # offset = (window size) / 2 - 0.5

        for y in range(height):
            for x in range(width):
                # get 1D pixel data from filter window and boundary-ness
                f_win, is_boundary = self.get_pixel_rel_win(f, y, x, ofs, flatten=True)
                # filter boundary if inc_bound = True
                if (not is_boundary) or (is_boundary and inc_bound):
                    # calculate index to median pixel
                    i_med = len(f_win) // 2
                    # sort, extract and assign median pixel data
                    g[y, x] = sorted(f_win)[i_med]
        
        if self.show: self.imshow(g, title='Filtered image')
        if self.verbose: print(f'Image is denoised with a median filter (size = {win_size})')

        return g

    def adaptive_median_filter(self, f, win_max):
        #
        # adaptive median filter
        #
        # f: input image data
        # win_max: maximum filter window size
        # g: filtered image data
        #
        g = np.zeros_like(f)
        height, width = f.shape
        S_max = win_max

        for y in range(height):
            for x in range(width):
                z_xy = f[y, x]
                for S in range(3, S_max + 1, 2): # S = 3, 5, ..., S_max
                    ofs = S // 2
                    f_win, _ = self.get_pixel_rel_win(f, y, x, ofs, flatten=True)
                    zs = np.array(sorted(f_win), dtype=np.int)
                    z_min = zs[0]
                    z_max = zs[-1]
                    z_med = zs[len(zs) // 2]
                    # Stage A
                    if z_min < z_med < z_max: # successfully filtered
                        # Stage B (check if not a noise)
                        z = z_xy if z_min < z_xy < z_max else z_med
                        break
                    else: # too many noises to filter with small window
                        z = z_med
                g[y, x] = z
        
        if self.show: self.imshow(g, title='Filtered image')
        if self.verbose: print(f'Image is denoised with an adaptive median filter (max. size = {S_max})')

        return g
