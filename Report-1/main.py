import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from ImgProc import MyImgProcLib
from utils import *

# parameters
filepath = 'Report-1/lenna.bmp'
# filepath = 'Report-1/prob.png'
ratio_salt = 0.25
ratio_pepper = 0.25
win_size = 7
max_size = 7

# create a MyImgProcLib object
ipl = MyImgProcLib(show=False, verbose=True)

# median filter test
f = ipl.load(filepath)
fn = ipl.add_salt_and_pepper(f, ratio_salt=ratio_salt, ratio_pepper=ratio_pepper)
g = ipl.median_filter(fn, win_size=win_size, inc_bound=True)
ga = ipl.adaptive_median_filter(fn, max_size=max_size)

p = [f, fn, ga, g]
title = [
    'original image',
    'salt-and-pepper',
    f'adaptive median filter(max={max_size})',
    f'median filter(size={win_size})',
    ]

fig, axis = plt.subplots(nrows=2, ncols=2*2)
axis = axis.flatten()
for i in range(len(p)):
    axis[i*2].imshow(p[i], cmap='gray')
    axis[i*2].set_title(title[i])
    axis[i*2].set_axis_off()

    hist = hist_fromarray(p[i])
    axis[i*2+1].plot(hist / sum(hist))
    axis[i*2+1].set_axis_off()
    axis[i*2+1].set_xlim(0, 255)
    axis[i*2+1].set_ylim(0, 0.02)
plt.show()
