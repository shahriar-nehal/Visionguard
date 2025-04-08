import numpy as np
import matplotlib.pyplot as plt

Figure_num = 0 #  or 1

num_scenes = 20

tp_num = np.array([0,0,0,0,0,0,1,1,3,8,10,14,18,19,20,20,20,20,20,20])
fp_num = np.array([20,20,20,16,16,9,4,3,2,0,0,0,0,0,0,0,0,0,0,0])

tp_num_2 = np.array([0,0,0,0,0,0,0,2,3,7,11,15,18,19,20,20,20,20,20,20])
fp_num_2 = np.array([20,20,20,20,15,11,8,5,3,2,1,0,0,0,0,0,0,0,0,0])

tp_num_3 = np.array([0,0,0,0,0,0,0,1,3,7,11,15,18,20,20,20,20,20,20,20])
fp_num_3 = np.array([20,20,20,20,16,11,8,5,3,2,0,0,0,0,0,0,0,0,0,0])

tp_num_4 = np.array([0,0,0,0,0,0,0,0,1,2,10,14,18,19,20,20,20,20,20,20])
fp_num_4 = np.array([20,20,19,19,16,12,9,6,3,1,0,0,0,0,0,0,0,0,0,0])

tp_num_5 = np.array([0,0,0,0,0,0,0,0,3,4,8,13,18,20,20,20,20,20,20,20])
fp_num_5 = np.array([20,20,19,19,16,12,9,6,3,1,0,0,0,0,0,0,0,0,0,0])

tp_num_6 = np.array([0,0,0,0,0,0,0,1,2,4,8,13,20,20,20,20,20,20,20,20]) # SLAP: weather
fp_num_6 = np.array([20,20,19,19,16,12,9,6,3,1,0,0,0,0,0,0,0,0,0,0])


tprs = tp_num/num_scenes
fprs = fp_num/num_scenes
tprs_2 = tp_num_2/num_scenes
fprs_2 = fp_num_2/num_scenes
tprs_3 = tp_num_3/num_scenes
fprs_3 = fp_num_3/num_scenes

tprs_4 = tp_num_4/num_scenes
fprs_4 = fp_num_4/num_scenes
tprs_5 = tp_num_5/num_scenes
fprs_5 = fp_num_5/num_scenes
tprs_6 = tp_num_6/num_scenes
fprs_6 = fp_num_6/num_scenes



# calculate average h
fprs_x = fprs_6
tprs_x = tprs_6


thresholds = np.linspace(0, 1, 20)

distances = (np.sqrt((fprs_x - 0)**2 + (tprs_x - 1)**2))
best_threshold_index = np.argmin(distances)
best_threshold = thresholds[best_threshold_index]

sorted_indices = np.argsort(fprs_x)
sorted_tprs = tprs_x[sorted_indices]
sorted_fprs = fprs_x[sorted_indices]

auc = np.trapz(sorted_tprs, sorted_fprs)

if Figure_num == 0:
    plt.plot(tprs, 1-fprs, linewidth=4, label='SLAP') #SLAP: ASPR
    plt.plot(tprs_2, 1-fprs_2, linewidth=4, label='RP2') #SLAP: ASPR
    plt.plot(tprs_3, 1-fprs_3, linewidth=4, label='ShapeS') # SLAP: ASPR
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis('tight')
    plt.axis('equal')
    plt.xlabel('False Positive', fontsize='x-large')
    plt.ylabel('True Positive', fontsize='x-large')
    # plt.title('')
    plt.legend(fontsize='x-large')
    plt.grid(True)
    plt.show()

else:

    plt.plot(tprs, 1-fprs, linewidth=4, label='Velocity') #SLAP: ASPR
    plt.plot(tprs_4, 1-fprs_4, linewidth=4, label='Heading') #SLAP: ASPR
    plt.plot(tprs_6, 1-fprs_6, linewidth=4, label='Weather') # SLAP: ASPR
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis('tight')
    plt.axis('equal')
    plt.xlabel('False Positive', fontsize='x-large')
    plt.ylabel('True Positive', fontsize='x-large')
    # plt.title('')
    plt.legend(fontsize='x-large')
    plt.grid(True)
    plt.show()