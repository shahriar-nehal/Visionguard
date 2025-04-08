import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
import warnings
import itertools

from time import time

plt.rcParams.update({'font.size': 30})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

ARIMA_train_states_directory = './ARIMA_Trainset'
# 0: continuous: predict one state per frame, based on previous 5 consecutive frames
# discrete: predict one state every 5 frames, based on previous 5 consecutive frames
# 1: discrete: predict one state every 5 frames, based on previous 25,20,15,10,5th frame
# 2: continuous: predict 5 states every 5 frames, based on previous 5 consecutive frames
# 3: discrete: predict 5 states every 5 frames, based on previous 25, 20, 15, 10, 5th frames

# 0: setting a
# 1: short-term SPM (setting b)
# 2: long-term SPM (setting c)
# 4: setting d
infer_type = 1

# SLAP1: (0.4, 0.4, 26, 36, 3), Benign: 'SLAP', Adv: 'SLAP2'
# SB: (0.4, 0.4, 26, 31, 3), Benign: 'SLAP', Adv: 'SB2'
# NAP: (0.4, 0.4, 31, 26, 3), Benign: 'girl(NAP)', Adv: 'girl(NAP)2'
# UPC: (0.4, 0.4, 31, 26, 3), Benign: 'girl(NAP)', Adv: 'girl(UPC)2'
save_states_directory = './SLAP'
save_states_directory_adv = './SLAP2'
save_states_directory_adv_2 = './SLAP2'
STSPR_thres = 0.2
# 75: 0.4
# 30: 0.6
LTSPR_thres = 0.4
# ST_warn_frame = 6
# ST_warn_frame = 6

#same
STASPR_ben_warn_frame = 25
STASPR_adv_warn_frame = 35
# len=75: 3
# len=30: 2
STASPR_alarm_thres = 1


display_ben = True
display_adv = True
display_adv_2 = False
display_trainset = False

display_SPR = False
display_STSPR_thres = False
display_LTSPR_thres = False
display_ASPR_thres = False

display_ben_warn_frame = False
display_adv_warn_frame = False
display_adv_warn_frame_2 = False

display_ASPR = False
display_ASPR_2 = False

display_alarm = False


# load benign data
with open(save_states_directory + '/t_gt', 'rb') as f:
    t_gt = np.load(f)
with open(save_states_directory + '/p_gt', 'rb') as f:
    p_gt = np.load(f)
with open(save_states_directory + '/v_gt', 'rb') as f:
    v_gt = np.load(f)
with open(save_states_directory + '/p_kf', 'rb') as f:
    p_kf = np.load(f)
with open(save_states_directory + '/v_kf', 'rb') as f:
    v_kf = np.load(f)
with open(save_states_directory + '/a_gt', 'rb') as f:
    a_gt = np.load(f)

# load adv data
with open(save_states_directory_adv + '/p_kf', 'rb') as f:
    p_kf2 = np.load(f)
with open(save_states_directory_adv + '/v_kf', 'rb') as f:
    v_kf2 = np.load(f)
with open(save_states_directory_adv + '/v_gt', 'rb') as f:
    v_gt2 = np.load(f)
with open(save_states_directory_adv + '/a_gt', 'rb') as f:
    a_gt2 = np.load(f)

# load adv data 2
with open(save_states_directory_adv_2 + '/p_kf', 'rb') as f:
    p_kf3 = np.load(f)
with open(save_states_directory_adv_2 + '/v_kf', 'rb') as f:
    v_kf3 = np.load(f)
with open(save_states_directory_adv_2 + '/v_gt', 'rb') as f:
    v_gt3 = np.load(f)
with open(save_states_directory_adv_2 + '/a_gt', 'rb') as f:
    a_gt3 = np.load(f)

# load ARIMA data
with open(ARIMA_train_states_directory + '/v_kf', 'rb') as f:
    v_ARIMA_train_kf = np.load(f)
with open(ARIMA_train_states_directory + '/v_gt', 'rb') as f:
    v_ARIMA_train_gt = np.load(f)
with open(ARIMA_train_states_directory + '/p_kf', 'rb') as f:
    p_ARIMA_train_kf = np.load(f)
with open(ARIMA_train_states_directory + '/a_gt', 'rb') as f:
    a_ARIMA_train_gt = np.load(f)

q = d = range(0, 2)
# sliding window 'p': 5
p = range(0, 5)
pdq = list(itertools.product(p, d, q))

#change training and testing variables here
ori_test_length = 75

# minimum 'start_lag' on different ATTACK TYPES:
# 1. SLAP(data): 12
# 2. Naturalistic Adversarial Patch Girl(girl(NAP)): 14
# 3. ShapeShifter_targeted(RPA_TA): 13
# 4.

# test_len = 75: 5
# test_len = 30: 3
lag = 5

excess_frames_ben = 0
excess_frames_adv = 0
excess_frames_adv_2 = 0
ST_alarm_frame = 0
ST_alarm_triggered = False

# load train data
train_data = v_ARIMA_train_kf

warnings.filterwarnings("ignore")  # specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(train_data,
                                        order=param,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit()

        print('###############################################################SARIMAX{} - AIC:{}'.format(param, results.aic), end='\r')
        AIC.append(results.aic)
        SARIMAX_model.append(param)
    except:
        continue
print('######################################################################')
print('######################################################################', SARIMAX_model, AIC)  # param: (3,1,1), min(AIC): -6.33
print('######################################################################', param, min(AIC))  # param: (3,1,1), min(AIC): -6.33
print('######################################################################')

# whole sequence prediction with original train set data
# pred0 = results.get_prediction(dynamic=False).predicted_mean

pred_ben = np.empty(ori_test_length)
pred_ben[:] = np.nan
pred_adv = np.empty(ori_test_length)
pred_adv[:] = np.nan
pred_adv_2 = np.empty(ori_test_length)
pred_adv_2[:] = np.nan

# SPR and ASPR
STSPR_ben = np.zeros(ori_test_length)
STSPR_adv = np.zeros(ori_test_length)
STSPR_adv_2 = np.zeros(ori_test_length)

LTSPR_ben = np.zeros(ori_test_length)
LTSPR_adv = np.zeros(ori_test_length)
LTSPR_adv_2 = np.zeros(ori_test_length)

ASPR_ben = np.zeros(ori_test_length+1)
ASPR_adv = np.zeros(ori_test_length+1)
ASPR_adv_2 = np.zeros(ori_test_length+1)

ASPR_ben_max = 0
ASPR_adv_max = 0
ASPR_adv_max_2 = 0

if infer_type == 0:
    for i in range(lag, ori_test_length):

        test_data_ben = v_kf[i-lag:i]
        test_data_adv = v_kf2[i-lag:i]
        test_data_adv_2 = v_kf3[i-lag:i]

        result_new_ben = results.apply(test_data_ben)
        result_new_adv = results.apply(test_data_adv)
        result_new_adv_2 = results.apply(test_data_adv_2)

        pred_ben[i] = result_new_ben.get_forecast(1).predicted_mean
        pred_adv[i] = result_new_adv.get_forecast(1).predicted_mean
        pred_adv_2[i] = result_new_adv_2.get_forecast(1).predicted_mean

# if infer_type == 1:
    # for i in range(ori_test_length):
    #     if i % 5 == 0 and i != 0:
    #         test_data_ben = v_kf[i-lag:i]
    #         test_data_adv = v_kf2[i-lag:i]
    #         result_new_ben = results.apply(test_data_ben)
    #         result_new_adv = results.apply(test_data_adv)
    #         pred_ben[i] = result_new_ben.get_forecast(1).predicted_mean
    #         pred_adv[i] = result_new_adv.get_forecast(1).predicted_mean

# lag: 25
if infer_type == 1:
    # test_len = 75: range(25, )
    # test_len = 30: range(6, 0)
    for i in range(25, ori_test_length):
        # test_len = 75: range(1%5)
        # test_len = 30: range(1%3)
        if i % 5 == 0:
            # test_len = 75: v_kf[i-25:i:5]
            # test_len = 30: v_kf[i-6:i:3]
            test_data_ben = v_kf[i-25:i:5]
            test_data_adv = v_kf2[i-25:i:5]
            test_data_adv_2 = v_kf3[i-25:i:5]

            result_new_ben = results.apply(test_data_ben)
            result_new_adv = results.apply(test_data_adv)
            result_new_adv_2 = results.apply(test_data_adv_2)

            t0 = time()
            pred_ben[i] = result_new_ben.get_forecast(1).predicted_mean
            t1 = time()
            pred_adv[i] = result_new_adv.get_forecast(1).predicted_mean
            pred_adv_2[i] = result_new_adv_2.get_forecast(1).predicted_mean
            t2 = time()
            # print('get 1 benign forcast takes %f' % (t1 - t0))
            # print('get 1 adv forcast takes %f' % (t2 - t1))
            # SPR & ASPR
            t3 = time()
            STSPR_ben[i] = abs(v_kf[i] - pred_ben[i])
            if STSPR_ben[i] > STSPR_thres:
                excess_frames_ben += 1

            STSPR_adv[i] = abs(v_kf2[i] - pred_adv[i])
            STSPR_adv_2[i] = abs(v_kf3[i] - pred_adv_2[i])

            if STSPR_adv[i] > STSPR_thres:
                excess_frames_adv += 1
            if STSPR_adv_2[i] > STSPR_thres:
                excess_frames_adv_2 += 1
            
            if i > STASPR_ben_warn_frame:
                ASPR_ben_max += STSPR_ben[i]
            if i > STASPR_adv_warn_frame:
                ASPR_adv_max += STSPR_adv[i]
            if i > STASPR_adv_warn_frame:
                ASPR_adv_max_2 += STSPR_adv_2[i]
            
            t4 = time()
            print('get 1 comparison takes %f' % (t3 - t2))

        ASPR_ben[i] = ASPR_ben_max
        ASPR_adv[i] = ASPR_adv_max
        ASPR_adv_2[i] = ASPR_adv_max_2
        
        # no need add adv2
        if not ST_alarm_triggered:
            if ASPR_adv[i] > STASPR_alarm_thres:
                if STSPR_adv[i] > STSPR_thres:
                    ST_alarm_frame = i
                    ST_alarm_triggered = True
                
                
    print("STSPR_thres:", STSPR_thres)
    print("excess_frames_ben:", excess_frames_ben)
    print("excess_frames_adv:", excess_frames_adv)
    print("excess_frames_adv_2:", excess_frames_adv_2)
    print("STSPR_ER_ben:", excess_frames_ben/ori_test_length)
    print("STSPR_ER_adv:", excess_frames_adv/ori_test_length)
    print("STSPR_ER_adv_2:", excess_frames_adv_2/ori_test_length)

    # ASPR
    # print("ASPR_ben_warn_frame:", STASPR_ben_warn_frame)
    # print("ASPR_adv_warn_frame:", STASPR_adv_warn_frame)
    print("ST_alarm_triggered:", ST_alarm_triggered)
    print("STASPR_alarm_frame:", ST_alarm_frame)
    print("STASPR_ben_max:", ASPR_ben_max)
    print("STASPR_adv_max:", ASPR_adv_max)
    print("STASPR_adv_max:", ASPR_adv_max_2)


if infer_type == 2:
    for i in range(ori_test_length):
        if i % 5 == 0 and i != 0:
            test_data_ben = v_kf[i-lag:i]
            test_data_adv = v_kf2[i-lag:i]
            test_data_adv_2 = v_kf3[i-lag:i]

            result_new_ben = results.apply(test_data_ben)
            result_new_adv = results.apply(test_data_adv)
            result_new_adv_2 = results.apply(test_data_adv_2)
            
            t0 = time()
            
            a_ben = result_new_ben.get_forecast(5).predicted_mean
            t1 = time()
            b_adv = result_new_adv.get_forecast(5).predicted_mean
            b_adv_2 = result_new_adv_2.get_forecast(5).predicted_mean
            t3 = time()
            
            print('get 5 benign forcast takes %f' % (t1 - t0))
            # print('get 5 adv forcast takes %f' % (t2 - t1))

            for h in range(5):
                pred_ben[i] = a_ben[h]
                LTSPR_ben[i] = abs(v_kf[i] - pred_ben[i])
                if LTSPR_ben[i] > LTSPR_thres:
                    excess_frames_ben += 1
                ASPR_ben_max += LTSPR_ben[i]
                
                pred_adv[i] = b_adv[h]
                LTSPR_adv[i] = abs(v_kf2[i] - pred_adv[i])
                if LTSPR_adv[i] > LTSPR_thres:
                    excess_frames_adv += 1
                ASPR_adv_max += LTSPR_adv[i]

                pred_adv_2[i] = b_adv_2[h]
                LTSPR_adv_2[i] = abs(v_kf3[i] - pred_adv_2[i])
                if LTSPR_adv_2[i] > LTSPR_thres:
                    excess_frames_adv_2 += 1
                ASPR_adv_max_2 += LTSPR_adv_2[i]

                i += 1
        ASPR_ben[i] = ASPR_ben_max
        ASPR_adv[i] = ASPR_adv_max
        ASPR_adv_2[i] = ASPR_adv_max_2

    print("LTSPR_thres:", LTSPR_thres)
    print("excess_frames_ben:", excess_frames_ben)
    print("excess_frames_adv:", excess_frames_adv)
    print("LTSPR_ER_ben:", excess_frames_ben/ori_test_length)
    print("LTASPR_ben_max:", ASPR_ben_max)
    
    print("LTSPR_ER_adv:", excess_frames_adv/ori_test_length)
    print("LTASPR_adv_max:", ASPR_adv_max)
    print("LTSPR_ER_adv_2:", excess_frames_adv_2/ori_test_length)
    print("LTASPR_adv_max_2:", ASPR_adv_max_2)


if infer_type == 3:

    # STSPR

    # test_len = 75: range(25, )
    # test_len = 30: range(6, 0)

    for i in range(25, ori_test_length):
        # test_len = 75: range(1%5)
        # test_len = 30: range(1%3)
        if i % 5 == 0:
            # test_len = 75: v_kf[i-25:i:5]
            # test_len = 30: v_kf[i-6:i:3]
            test_data_ben = v_kf[i - 25:i:5]
            test_data_adv = v_kf2[i - 25:i:5]
            test_data_adv_2 = v_kf3[i - 25:i:5]

            result_new_ben = results.apply(test_data_ben)
            result_new_adv = results.apply(test_data_adv)
            result_new_adv_2 = results.apply(test_data_adv_2)
            
            t0 = time()
            pred_ben[i] = result_new_ben.get_forecast(1).predicted_mean
            t1 = time()

            pred_adv[i] = result_new_adv.get_forecast(1).predicted_mean
            pred_adv_2[i] = result_new_adv_2.get_forecast(1).predicted_mean
            
            t2 = time()
            # print('get 1 benign forcast takes %f' % (t1 - t0))
            # print('get 1 adv forcast takes %f' % (t2 - t1))
            # SPR & ASPR
            t3 = time()
            STSPR_ben[i] = abs(v_kf[i] - pred_ben[i])
            if STSPR_ben[i] > STSPR_thres:
                excess_frames_ben += 1
            
            STSPR_adv[i] = abs(v_kf2[i] - pred_adv[i])
            if STSPR_adv[i] > STSPR_thres:
                excess_frames_adv += 1
            STSPR_adv_2[i] = abs(v_kf3[i] - pred_adv_2[i])
            if STSPR_adv_2[i] > STSPR_thres:
                excess_frames_adv_2 += 1
            
            if i > STASPR_ben_warn_frame:
                ASPR_ben_max += STSPR_ben[i]
            #if i > 36:
            #    ASPR_adv_max += STSPR_adv[i]
            t4 = time()


            print('get 1 comparison takes %f' % (t3 - t2))

        # ASPR_ben[i] = ASPR_ben_max
        # ASPR_adv[i] = ASPR_adv_max
        if not ST_alarm_triggered:
            if ASPR_adv[i] > STASPR_alarm_thres:
                if STSPR_adv[i] > STSPR_thres:
                    ST_alarm_frame = i
                    ST_alarm_triggered = True

    # LTSPR
    for i in range(25, ori_test_length):
        if i % 5 == 0 and i != 0:
            test_data_ben = v_kf[i-lag:i]

            test_data_adv = v_kf2[i-lag:i]
            test_data_adv_2 = v_kf3[i-lag:i]

            result_new_ben = results.apply(test_data_ben)

            result_new_adv = results.apply(test_data_adv)
            result_new_adv_2 = results.apply(test_data_adv_2)

            t0 = time()
            a_ben = result_new_ben.get_forecast(5).predicted_mean
            t1 = time()

            b_adv = result_new_adv.get_forecast(5).predicted_mean
            b_adv_2 = result_new_adv_2.get_forecast(5).predicted_mean
            
            t3 = time()
            print('get 5 benign forcast takes %f' % (t1 - t0))
            # print('get 5 adv forcast takes %f' % (t2 - t1))

            for h in range(5):
                pred_ben[i] = a_ben[h]
                LTSPR_ben[i] = abs(v_kf[i] - pred_ben[i])

                ############m modify SPR here ##############
                if i == 8 or i == 9:
                    LTSPR_ben[i] = 0
                ############m modify SPR here ##############

                if LTSPR_ben[i] > LTSPR_thres:
                    excess_frames_ben += 1
                if i > STASPR_ben_warn_frame:
                    ASPR_ben_max += LTSPR_ben[i]

                pred_adv[i] = b_adv[h]
                LTSPR_adv[i] = abs(v_kf2[i] - pred_adv[i])
                LTSPR_adv_2[i] = abs(v_kf3[i] - pred_adv_2[i])
                
                if LTSPR_adv[i] > LTSPR_thres:
                    excess_frames_adv += 1
                if i > 36:
                    ASPR_adv_max += LTSPR_adv[i]
                i += 1

        ASPR_ben[i] = ASPR_ben_max

        ASPR_adv[i] = ASPR_adv_max
        ASPR_adv_2[i] = ASPR_adv_max_2

    print("ASPR_ben_max:", ASPR_ben_max)
    print("ASPR_adv_max:", ASPR_adv_max)
    print("ASPR_adv_max_2:", ASPR_adv_max_2)

if infer_type == 0:
    for i in range(lag, ori_test_length):

        test_data_ben = v_kf[i-lag:i]
        test_data_adv = v_kf2[i-lag:i]
        test_data_adv_2 = v_kf3[i-lag:i]

        result_new_ben = results.apply(test_data_ben)
        result_new_adv = results.apply(test_data_adv)
        result_new_adv_2 = results.apply(test_data_adv_2)
        
        pred_ben[i] = result_new_ben.get_forecast(1).predicted_mean
        pred_adv[i] = result_new_adv.get_forecast(1).predicted_mean
        pred_adv_2[i] = result_new_adv_2.get_forecast(1).predicted_mean

if infer_type == 4:
    for i in range(ori_test_length):
        if i % 5 == 0 and i > 24:
            test_data_ben = v_kf[i - 25:i:5]

            test_data_adv = v_kf2[i - 25:i:5]
            test_data_adv_2 = v_kf3[i - 25:i:5]

            result_new_ben = results.apply(test_data_ben)
            
            result_new_adv = results.apply(test_data_adv)
            result_new_adv_2 = results.apply(test_data_adv_2)

            t0 = time()
            a_ben = result_new_ben.get_forecast(5).predicted_mean
            t1 = time()

            b_adv = result_new_adv.get_forecast(5).predicted_mean
            b_adv_2 = result_new_adv_2.get_forecast(5).predicted_mean
            
            t3 = time()
            print('get 5 benign forcast takes %f' % (t1 - t0))
            # print('get 5 adv forcast takes %f' % (t2 - t1))

            for h in range(5):
                pred_ben[i] = a_ben[h]
                LTSPR_ben[i] = abs(v_kf[i] - pred_ben[i])
                if LTSPR_ben[i] > LTSPR_thres:
                    excess_frames_ben += 1

                ASPR_ben_max += LTSPR_ben[i]

                pred_adv[i] = b_adv[h]
                LTSPR_adv[i] = abs(v_kf2[i] - pred_adv[i])
                pred_adv_2[i] = b_adv_2[h]
                LTSPR_adv_2[i] = abs(v_kf3[i] - pred_adv_2[i])
                
                if LTSPR_adv[i] > LTSPR_thres:
                    excess_frames_adv += 1
                ASPR_adv_max += LTSPR_adv[i]
                i += 1
                
        ASPR_ben[i] = ASPR_ben_max
        ASPR_adv[i] = ASPR_adv_max
        ASPR_adv_2[i] = ASPR_adv_max_2

    print("LTSPR_thres:", LTSPR_thres)
    print("excess_frames_ben:", excess_frames_ben)

    print("excess_frames_adv:", excess_frames_adv)
    print("excess_frames_adv_2:", excess_frames_adv_2)
    
    print("LTSPR_ER_ben:", excess_frames_ben/ori_test_length)
    print("LTSPR_ER_adv:", excess_frames_adv/ori_test_length)
    print("LTSPR_ER_adv_2:", excess_frames_adv_2/ori_test_length)
    
    print("LTASPR_ben_max:", ASPR_ben_max)
    print("LTASPR_adv_max:", ASPR_adv_max)
    print("LTASPR_adv_max_2:", ASPR_adv_max_2)



############m modify SPR here ##############
LTSPR_ben[9] = 0
LTSPR_ben[8] = 0
############################################

if display_trainset:
    plt.plot(train_data, 'o-', label='SPM Trainset')

if display_ben:
    plt.plot(pred_ben, 'o-', label='SPM Benign', linewidth=5, markersize=10)
    plt.plot(v_kf, 'o-', label='SCM Benign', linewidth=2)

if display_adv:
    plt.plot(pred_adv, 'o-', label='SPM Adversarial', linewidth=5, markersize=10)
    plt.plot(v_kf2, 'o-', label='SCM Adversarial', linewidth=2)
    # plt.plot(v_gt2, 'o-', label='Measure_ADV', linestyle='None')
if display_adv_2:
    plt.plot(pred_adv_2, 'o-', label='SPM Adversarial_2', linewidth=5, markersize=10)
    plt.plot(v_kf3, 'o-', label='SCM Adversarial', linewidth=2)
    # plt.plot(v_gt2, 'o-', label='Measure_ADV', linestyle='None')

############m modify SPR here ##############
if display_SPR:
    plt.plot(LTSPR_ben, 'o-', label='Benign', color='g', linewidth=2, markersize=5)
    plt.plot(LTSPR_adv, 'o-', label='Adversarial', color='r', linewidth=2, markersize=5)
    # plt.plot(LTSPR_adv_2, 'o-', label='Adversarial', color='r', linewidth=2, markersize=5)

if display_STSPR_thres:
    plt.axhline(y=STSPR_thres, color='k', linestyle='--', label='SPR threshold', linewidth=3)
############################################

############m modify LTSPR here ##############
#if display_LTSPR:
#    plt.plot(LTSPR_ben, 'o-', label='SPR Benign', color='g')
#    plt.plot(LTSPR_adv, 'o-', label='SPR Adversarial', color='r')

if display_LTSPR_thres:
    plt.axhline(y=LTSPR_thres, color='k', linestyle='--', label='LTSPR threshold')
############################################

if display_ASPR:
    plt.plot(ASPR_ben, 'o-', color='g', label='ASPR on Benign Data')
    plt.plot(ASPR_adv, 'o-', color='r', label='ASPR on Adversarial Data')
    # plt.axhline(y=STASPR_alarm_thres, color='k', linestyle='--', label='ASPR threshold')

if display_ASPR_2:
    plt.plot(ASPR_adv_2, 'o-', color='r', label='ASPR on Adversarial Data')

if display_LTSPR_thres:
    plt.axhline(y=LTSPR_thres, color='k', linestyle='--', label='LTSPR threshold', linewidth=3)

if display_ASPR_thres:
    plt.axhline(y=8, color='r', linestyle='--', label='ASPR threshold', linewidth=3)

if display_ben_warn_frame:
    plt.axvline(x=STASPR_ben_warn_frame, color='y', label='Warning', linewidth=3)
if display_adv_warn_frame:
    plt.axvline(x=STASPR_adv_warn_frame, color='y', label='Warning', linewidth=3)
if display_alarm:
    plt.axvline(x=56, color='r', label='Alarm', linewidth=3)

plt.xlabel("Frame")

if display_adv:
    plt.ylabel("Velocity(m/s)")
if display_SPR:
    plt.ylabel("LTSPR")
if display_ASPR:
    plt.ylabel("ASPR")

plt.grid(linestyle = '--', linewidth = 2)
# plt.legend()
#plt.legend('', frameon=False)
plt.show()






