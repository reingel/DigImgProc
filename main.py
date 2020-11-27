import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from ImgProc import ImgProcLib

# parameters
filepath = 'camera.bmp'

# create a ImgProcLib object
ipl = ImgProcLib(show=False, verbose=True)

# load image file
f = ipl.load(filepath)

# calculate probability distribution
p = ipl.prob_dist(f)
g, p, sB2, k_star = ipl.threshold_otsu(f)

plt.subplot(221)
plt.imshow(f, cmap='gray')

plt.subplot(222)
plt.imshow(g, cmap='gray')

plt.subplot(223)
plt.plot(p)
plt.plot(sB2/np.max(sB2))
plt.plot([k_star, k_star], [0, 1])

plt.show()


'''
p = [f, fn, None] # image array
t = [f'original image (diff={d:.2f})', f'corrupted by salt-and-pepper (diff={dn:.2f})', None] # title array
for s in range(3,7+1,2):
    g = ipl.median_filter(fn, win_size=s)
    h = ipl.difference(f, g)
    d = np.sqrt(np.mean(h**2))
    p.append(g)
    t.append(f'median filter (size={s}, diff={d:.2f})')
for s in range(3,7+1,2):
    g = ipl.adaptive_median_filter(fn, win_max=s)
    h = ipl.difference(f, g)
    d = np.sqrt(np.mean(h**2))
    p.append(g)
    t.append(f'adaptive median filter (max. size={s}, diff={d:.2f})')

fig, axis = plt.subplots(nrows=3, ncols=3)
axis = axis.flatten()
for i in range(len(p)):
    if np.array(p[i]).any():
        axis[i].imshow(p[i], cmap='gray')
        axis[i].set_title(t[i], fontsize=7)
    axis[i].set_axis_off()
plt.show()
'''