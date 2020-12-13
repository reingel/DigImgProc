import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ImgProc import ImgProcLib

nBit = np.int(8) # number of bits for gray levels
L = np.int(2**nBit) # number of gray levels
IMIN = np.uint8(0) # black
IMAX = np.uint8(L - 1) # white

# image filenames
filenames = [
    'Fig2.07(a).jpg',
    'A_1.png',
    'A_2.png',
    'mesh.png',
    'honeywell.png',
    'stain_1.png',
    'stain_2.png',
    'script1.jpeg',
    'script2.jpeg',
    'Fig2.08(a).jpg',
    'Fig2.08(b).jpg',
    'Fig2.08(c).jpg',
    'Fig2.19(a).jpg',
    'Fig2.21(a).jpg',
    'Fig2.22(a).jpg',
    'Fig2.22(b).jpg',
    'Fig2.22(c).jpg',
    # 'Fig2.24.jpg',
    'Fig5.07(a).jpg',
    # 'Fig10.1(a).png',
    # 'Fig10.16(a).png',
    'Fig10.21(b).png',
    # 'Fig10.26(a).png',
    'Fig10.36(a).png',
    'Fig10.36(b).png',
    'Fig10.36(c).png',
    'Fig10.37(a).png',
    # 'Fig10.37(b).png',
    'Fig10.37(c).png',
    'Fig10.38(a).png',
    # 'Fig10.39(a).png',
    'Fig10.43(a).png',
    'Fig10.45(a).png',
    'Fig10.49(a).png',
    'Fig10.50(a).png',
]

# create a ImgProcLib object
ipl = ImgProcLib(show=False, verbose=False)

for filename in filenames:
    # load image file
    f = ipl.load('images/' + filename)
    
    # show the original image
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10,8))

    ax1.imshow(f, cmap='gray')
    ax1.set_title(f'Original image: {filename}', fontweight='bold', fontsize=8)
    ax1.set_axis_off()

    # Otsu's thresholding with two classes
    g, p, w_k, mu_k, mu_T, var_B, var_T, k_opt, var_B_opt, eta_opt, w_opt, mu_opt = ipl.threshold_otsu2(f)
    sigma_B = np.sqrt(var_B)
    sigma_T = np.sqrt(var_T)
    sigma_B_opt = np.sqrt(var_B_opt)
    k_opt2 = k_opt

    # show the segmented image
    ax2.imshow(g, cmap='gray')
    ax2.set_title("Otsu's thresholding with two classes", fontweight='bold', fontsize=8)
    ax2.set_axis_off()

    # show the histogram and class means
    ax5.vlines(range(0, L, 4), 0, p[::4], 'k', linewidth=0.5)
    # ax5.plot(w_k, 'r')
    ax5.vlines(k_opt, 0, np.max(p), 'r')
    ax5.vlines(mu_T, 0, np.max(p) * 2/3, 'c')
    ax5.vlines(mu_opt, 0, np.max(p) * 1/2, 'b')
    ax5.tick_params(axis='both', labelsize=7)
    ax5.set_xlabel('gray-level intensity', fontsize=7)
    ax5.set_ylabel('probability', fontsize=7)

    # show the between-class variance and optimal threshold
    ax8.plot(sigma_B, 'g', linewidth=0.5)
    ax8.plot([k_opt, k_opt], [0, sigma_B_opt], 'r')
    ax8.tick_params(axis='both', labelsize=7)
    ax8.set_xlabel('gray-level intensity', fontsize=7)
    ax8.set_ylabel('between-class std', fontsize=7)
    
    # show information about total data
    ax4.text(0, 0.8, 'Total mean & std', fontweight='bold', fontsize=8)
    ax4.text(0, 0.7, f'$\mu_T$ = {mu_T:.1f}, $\sigma_T$ = {sigma_T:.1f}', fontsize=8)

    # show information about two-class thresholding
    ax4.text(0, 0.5, 'two-class thresholding', fontweight='bold', fontsize=8)
    ax4.text(0, 0.4, f'$k^*$ = {k_opt}', fontsize=8)
    ax4.text(0, 0.3, f'$\sigma_B^*$ = {sigma_B_opt:.1f}, $\eta^*$ = {eta_opt:.3f}', fontsize=8)
    ax4.text(0, 0.2, f'$\omega^*$ = ({w_opt[0]:.2f}, {w_opt[1]:.2f})', fontsize=8)
    ax4.text(0, 0.1, f'$\mu^*$ = ({mu_opt[0]:.1f}, {mu_opt[1]:.1f})', fontsize=8)
    ax4.set_axis_off()

    # Otsu's thresholding with three classes
    g, p, w_k, mu_k, mu_T, var_B, var_T, k_opt, var_B_opt, eta_opt, w_opt, mu_opt = ipl.threshold_otsu3(f)
    sigma_B = np.sqrt(var_B)
    sigma_T = np.sqrt(var_T)
    sigma_B_opt = np.sqrt(var_B_opt)

    # show the segmented image
    ax3.imshow(g, cmap='gray')
    ax3.set_title("Otsu's thresholding with three classes", fontweight='bold', fontsize=8)
    ax3.set_axis_off()

    # show the histogram and class means
    ax6.vlines(range(0, L, 4), 0, p[::4], 'k', linewidth=0.5)
    # ax6.plot(mu_k, 'r')
    ax6.vlines(k_opt, 0, np.max(p), 'r')
    ax6.vlines(mu_T, 0, np.max(p) * 2/3, 'c')
    ax6.vlines(mu_opt, 0, np.max(p) * 1/2, 'b')
    ax6.tick_params(axis='both', labelsize=7)
    ax6.set_xlabel('gray-level intensity', fontsize=7)
    ax6.set_ylabel('probability', fontsize=7)

    # show the between-class variance and optimal thresholds
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    ax9.contour(x, y, np.transpose(var_B), np.linspace(1, np.max(var_B), 50), linewidths=0.5)
    ax9.plot([0, k_opt[0], k_opt[0]], [k_opt[1], k_opt[1], 0], 'r')
    ax9.plot([0, 20], [k_opt2, k_opt2], 'r')
    ax9.plot([k_opt2, k_opt2], [235, 255], 'r')
    ax9.tick_params(axis='both', labelsize=7)
    ax9.set_xlabel('optimal threshold 1', fontsize=7)
    ax9.set_ylabel('optimal threshold 2', fontsize=7)
    
    # show information about three-class thresholding
    ax7.text(0, 0.8, 'three-class thresholding', fontweight='bold', fontsize=8)
    ax7.text(0, 0.7, f'$k^*$ = {k_opt}', fontsize=8)
    ax7.text(0, 0.6, f'$\sigma_B^*$ = {sigma_B_opt:.1f}, $\eta^*$ = {eta_opt:.3f}', fontsize=8)
    ax7.text(0, 0.5, f'$\omega^*$ = ({w_opt[0]:.2f}, {w_opt[1]:.2f}, {w_opt[2]:.2f})', fontsize=8)
    ax7.text(0, 0.4, f'$\mu^*$ = ({mu_opt[0]:.1f}, {mu_opt[1]:.1f}, {mu_opt[2]:.1f})', fontsize=8)
    ax7.set_axis_off()

    # thresholding using moving averages
    # if filename == 'Fig10.49(a).png' or filename == 'Fig10.50(a).png':
    #     g, t = ipl.threshold_moving_average(f, n=20, b=0.8)
    #     fig = plt.figure(figsize=(10/3,8/3*2))
    #     ax = fig.add_subplot(211)
    #     ax.imshow(t, cmap='gray')
    #     ax.set_axis_off()
    #     ax.set_title('moving average', fontweight='bold', fontsize=8)
    #     ax = fig.add_subplot(212)
    #     ax.imshow(g, cmap='gray')
    #     ax.set_axis_off()
    #     ax.set_title('thresholding', fontweight='bold', fontsize=8)

plt.show()
