import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from PIL import Image

def show_fromarray(p):
    img = Image.fromarray(p)
    img.show(img)
    
    return img

def hist_fromarray(p):
    img = Image.fromarray(p)
    return np.array(img.histogram())