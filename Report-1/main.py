import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from ImgProc import MyImgProcLib

# parameters
filepath = 'Report-1/lenna.bmp'
ratio_salt = 0.05
ratio_pepper = 0.05
size_filter = 5

# create a MyImgProcLib object
ipl = MyImgProcLib(show=True, verbose=True)

# median filter test
p = ipl.load(filepath)
pn = ipl.add_salt_and_pepper(p, ratio_salt=0.05, ratio_pepper=0.05)
po = ipl.median_filter(pn, window_size=5)
