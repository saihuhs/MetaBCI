from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional
from mne.filter import resample,notch_filter
import matplotlib.pyplot as plt
from collections import OrderedDict
from numpy import ndarray
from scipy.signal import sosfiltfilt
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from metabci.brainda.algorithms.decomposition.cca import _scca_feature,_ged_wong,_scca_kernel
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices, match_loo_indices)
from metabci.brainda.algorithms.decomposition import (SCCA,
    generate_filterbank, generate_cca_references)
from joblib import Parallel, delayed
import numpy as np
from mne import find_events, events_from_annotations
from mne.io import Raw
from scipy.signal import resample
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from sklearn.model_selection import train_test_split
from functools import partial
from metabci.brainda.algorithms.decomposition.cca import SCCA_LDA,SCCA_LDA_Multi_Subjects

from fif_dataset import Test1#这里为导入的dataset

#数据集与参数定义
dataset = Test1()
delay = 0.14 # 信号延迟时间seconds
channels = ['TP10','O2','Oz','O1','POz','Pz','TP9','FCz']
srate = 500# Hz
duration = 1 # seconds信号持续时间
n_harmonics = 2 #各频带的谐波数量
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]  #事件对应频率
phases = [dataset.get_phase(event) for event in events]  #事件相位信息
start_pnt = dataset.events[events[0]][1][0]
#生成参考信号
Yf = generate_cca_references(
    freqs, srate, duration, 
    phases=None, 
    n_harmonics=n_harmonics)


paradigm = SSVEP(
    srate=srate,
    channels=channels,
    intervals=[(start_pnt+delay, start_pnt+delay+duration)],
    events=events)#创建范式
#滤波
def raw_hook(raw, caches):
   
    # Apply notch filter at 50 Hz
    raw.notch_filter(50, picks='eeg', method='fir', fir_window='hamming', fir_design='firwin')
    # do something with raw object
    raw.filter(8, 55, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')#经过两次零相位滤波，第一次正向，第二次反向，这样可以消除相位失真。
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1#跟踪处理流程的进度或阶段。
    # 降采样
    raw.resample(250)  # 直接在raw对象上调用resample
    return raw, caches

paradigm.register_raw_hook(raw_hook)

set_random_seeds(64)#设置随机种子，保证可重复性

subjects = [1]
model = 1
acc = SCCA_LDA_Multi_Subjects(subjects=subjects,model = model,paradigm = paradigm,dataset= dataset,Yf=Yf,srate=srate,duration=duration)
print("SCCA_LDA for Model{} Acc:{:.4f}".format(model, acc))
