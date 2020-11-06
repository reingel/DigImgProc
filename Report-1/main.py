import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from ImgProc import ImgProcLib
from utils import *

# parameters
filepath_org = 'Report-1/Fig5.07(a).jpg'
filepath_pep = 'Report-1/Fig5.08(a).jpg'
filepath_sal = 'Report-1/Fig5.08(b).jpg'
filepath_snp10 = 'Report-1/Fig5.10(a).jpg'
filepath_snp25 = 'Report-1/Fig5.14(a).jpg'
# filepath = 'Report-1/lenna.bmp'
ratio_salt = 0.25
ratio_pepper = 0.25

# create a MyImgProcLib object
ipl = ImgProcLib(show=False, verbose=True)

# median filter test
f = ipl.load(filepath_org)
h = ipl.difference(f, f)
d = np.sqrt(np.mean(h**2))

fn = ipl.add_salt_and_pepper(f, ratio_salt=ratio_salt, ratio_pepper=ratio_pepper)
h = ipl.difference(f, fn)
dn = np.sqrt(np.mean(h**2))

p = [f, fn, None] # image array
q = [0, 0, None] # rms of difference array
t = [f'original image (diff={d:.2f})', f'corrupted by salt-and-pepper (diff={dn:.2f})', None] # title array
for s in range(3,7+1,2):
    g = ipl.median_filter(fn, win_size=s)
    h = ipl.difference(f, g)
    d = np.sqrt(np.mean(h**2))
    p.append(g)
    q.append(d)
    t.append(f'median filter (size={s}, diff={d:.2f})')
for s in range(3,7+1,2):
    g = ipl.adaptive_median_filter(fn, win_max=s)
    h = ipl.difference(f, g)
    d = np.sqrt(np.mean(h**2))
    p.append(g)
    q.append(d)
    t.append(f'adaptive median filter (max. size={s}, diff={d:.2f})')

fig, axis = plt.subplots(nrows=3, ncols=3)
axis = axis.flatten()
for i in range(len(p)):
    if np.array(p[i]).any():
        axis[i].imshow(p[i], cmap='gray')
        axis[i].set_title(t[i], fontsize=7)
    axis[i].set_axis_off()

    # hist = hist_fromarray(p[i])
    # axis[i*2+1].plot(hist / sum(hist))
    # axis[i*2+1].set_axis_off()
    # axis[i*2+1].set_xlim(0, 255)
    # axis[i*2+1].set_ylim(0, 0.02)
plt.show()
