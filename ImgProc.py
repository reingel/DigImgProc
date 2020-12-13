import numpy as np
import numpy.random as rd
from PIL import Image

nBit = np.int(8) # number of bits for gray levels
L = np.int(2**nBit) # number of gray levels
IMIN = np.uint8(0) # black
IMAX = np.uint8(L - 1) # white

class ImgProcLib:
    def __init__(self, show=False, verbose=True):
        self.show = show
        self.verbose = verbose

    def load(self, filepath):
        # load an image
        img = Image.open(filepath)

        # get data & info
        data = np.array(img)
        width, height = img.size
        mode = 'B&W' if img.mode == 'L' else img.mode

        # convert into a numpy array
        if mode == 'B&W':
            f = np.array(data, dtype=np.uint8)
        elif mode == 'RGB' or mode == 'RGBA':
            f = np.array(np.mean(data[:,:,:3], axis=2), dtype=np.uint8)
        else:
            print('Unknown mode!')
            return

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
        h = np.zeros(L, dtype=np.int)
        for s in f.flatten():
            h[s] += 1
        return h
    
    def prob_dist(self, f):
        N = len(f.flatten())
        h = self.histogram(f)
        return h / N

    def threshold_otsu2(self, f):
        i = range(L)
        p = self.prob_dist(f)
        w_k = np.cumsum(p) # class probability vector
        mu_k = np.cumsum(i * p) # class mean vector
        mu_T = mu_k[-1] # total mean
        var_B = np.zeros_like(w_k)
        for k in range(1,L):
            if 0 < w_k[k] < 1:
                var_B[k] = (mu_T * w_k[k] - mu_k[k])**2 / (w_k[k] * (1 - w_k[k]))
        var_T = np.sum((i - mu_T)**2 * p)
        # get optimal intensity
        k_opt = np.argmax(var_B)
        # extract the statistical information
        var_B_opt = var_B[k_opt]
        eta_opt = var_B_opt / var_T
        w_0 = w_k[k_opt]
        w_1 = 1 - w_0
        mu_0 = mu_k[k_opt] / w_0
        mu_1 = (mu_T - mu_0 * w_0) / (1 - w_0)
        # generate binary image
        g = np.zeros_like(f)
        g[f >= k_opt] = IMAX

        if self.verbose:
            print(f'{mu_T = :.1f}, sigma_T = {np.sqrt(var_T):.1f}')
            print(f'{k_opt = }, {eta_opt = :.3f}')
            print(f'{w_0 = :.3f}, {w_1 = :.3f}')
            print(f'{mu_0 = :.1f}, {mu_1 = :.1f}')
        
        return g, p, w_k, mu_k, mu_T, var_B, var_T, (k_opt), var_B_opt, eta_opt, (w_0, w_1), (mu_0, mu_1)

    def threshold_otsu3(self, f):
        i = range(L)
        p = self.prob_dist(f)
        w_k = np.cumsum(p) # class probability vector
        mu_k = np.cumsum(i * p) # class mean vector
        mu_T = mu_k[-1] # total mean
        var_B = np.zeros((L,L))
        for k1 in range(1,L-1): # k1 = 0, 1, ..., L - 2
            for k2 in range(k1+1, L): # k2 = k1 + 1, k1 + 2, ..., L - 1
                w_k1 = w_k[k1]
                w_k2 = w_k[k2]
                mu_k1 = mu_k[k1]
                mu_k2 = mu_k[k2]
                w = np.array([w_k1, w_k2 - w_k1, 1 - w_k2])
                if np.min(w**2) > 0:
                        mu = np.array([mu_k1, mu_k2 - mu_k1, mu_T - mu_k2]) / w
                        var_B[k1][k2] = np.sum(w * (mu - mu_T)**2)
        var_T = np.sum((i - mu_T)**2 * p)
        # get optimal intensity (axis=None means the index is into the flattened array)
        k1_opt, k2_opt = np.unravel_index(np.argmax(var_B, axis=None), var_B.shape)
        # extract the statistical information
        var_B_opt = var_B[k1_opt][k2_opt]
        eta_opt = var_B_opt / var_T
        w_0 = w_k[k1_opt]
        w_1 = w_k[k2_opt] - w_0
        w_2 = 1 - w_k[k2_opt]
        mu_0 = mu_k[k1_opt] / w_0
        mu_1 = (mu_k[k2_opt] - mu_k[k1_opt]) / w_1
        mu_2 = (mu_T - mu_k[k2_opt]) / w_2
        # generate binary image
        g = np.zeros_like(f)
        g[(k1_opt <= f) & (f < k2_opt)] = IMAX / 2
        g[k2_opt <= f] = IMAX

        if self.verbose:
            print(f'{mu_T = :.1f}, sigma_T = {np.sqrt(var_T):.1f}')
            print(f'{k1_opt = }, {k2_opt = }, {eta_opt = :.3f}')
            print(f'{w_0 = :.3f}, {w_1 = :.3f}, {w_2 = :.3f}')
            print(f'{mu_0 = :.1f}, {mu_1 = :.1f}, {mu_2 = :.1f}')
        
        return g, p, w_k, mu_k, mu_T, var_B, var_T, (k1_opt, k2_opt), var_B_opt, eta_opt, (w_0, w_1, w_2), (mu_0, mu_1, mu_2)

    def threshold_moving_average(self, f, n=5, b=0.5):
        height, width = f.shape
        z = np.array(f.flatten(), dtype=np.int)
        N = len(z)
        m = np.zeros_like(z)
        for k in range(N-1): # k = 0, 1, ..., N-2
            zkp1 = z[k+1]
            zkmn = z[k-n] if k >= n else 0
            m[k+1] = m[k] + (zkp1 - zkmn) / n
        t = np.reshape(m, f.shape)
        g = np.zeros_like(f)
        # local thresholding
        for y in range(height):
            for x in range(width):
                Sxy, _ = self.get_pixel_rel_win(t, y, x, 5, True)
                mxy = np.mean(Sxy)
                g[y, x] = 1 if f[y, x] > b * mxy else 0

        return g, t


    #
    # Report 1
    #
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
