# -*- coding: utf-8 -*-
from psychopy import monitors
import numpy as np

from metabci.brainstim.framework import Experiment
from metabci.brainstim.paradigm import paradigm,SSVEP
from metabci.brainflow.amplifiers import DataAcquisition
from metabci.brainstim.utils import NeuraclePort,Niantong_port 
import time
import mne
from mne.filter import resample
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_cca_references
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_fif
from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from joblib import Parallel, delayed
from functools import partial
from typing import Optional
from numpy import ndarray
from metabci.brainda.algorithms.decomposition.cca import _scca_feature
import threading
import queue

class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.lda_ = LinearDiscriminantAnalysis()

    def fit(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        Yf: Optional[ndarray] = None,
    ):

        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
       
        rhos = self.transform(X)
        self.lda_.fit(rhos, y)

        return self

    def transform(self, X: ndarray):
        try:
            X = np.reshape(X, (-1, *X.shape[-2:]))
            X = X - np.mean(X, axis=-1, keepdims=True)
            print('X shape:',X.shape)
            Yf = self.Yf_
            n_components = self.n_components
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_scca_feature, n_components=n_components))(a, Yf) for a in X
            )
            rhos = np.stack(rhos)
            return rhos
        except Exception as e:
            print('transform error:',e)

    def predict(self, X):
        try:
             if isinstance(X, list):
                 rhos_ = []
                 for i in X:
                     rhos = self.transform(i)
                     print('length rhos0', rhos.shape)
                     rhos_.append(rhos)
     
                 if len(rhos_) < 2:
                     labels = self.lda_.predict(rhos)
     
                 # 选择最大相关系数
                 new_rhos = np.maximum.reduce(rhos_)
     
                 # 使用训练好的 LDA 分类器进行预测
                 labels = self.lda_.predict(new_rhos)
             else:
                 rhos = self.transform(X)
                 print('length rhos2', rhos.shape)
                 # 使用训练好的 LDA 分类器进行预测
                 labels = self.lda_.predict(rhos)
             return labels
        except Exception as e:
             print(f"An error occurred during prediction: {e}")

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

def read_data(run_files, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_fif(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        events = mne.find_events(raw, stim_channel='TRIGGER')
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events,
                            event_id=labels,
                            tmin=interval[0],
                            tmax=interval[1],
                            baseline=None,
                            picks=ch_picks,
                            verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    return Xs, ys, ch_picks

def train_model(X, y, srate=500):
    if not isinstance(X, np.ndarray):
        raise ValueError("X should be a NumPy array.")
    y = np.reshape(y, (-1))
    X = resample(X, up=250, down=srate)

    X = mne.filter.notch_filter(X, Fs=srate, freqs=50, picks=np.arange(X.shape[1]), method='fir', fir_window='hamming',
                                fir_design='firwin', verbose=False)

    X = mne.filter.filter_data(X, sfreq=srate, l_freq=8, h_freq=55, l_trans_bandwidth=2, h_trans_bandwidth=5,
                               method='fir', phase='zero-double', verbose=False)

    freqs = np.arange(8, 16, 0.4)
    Yf = generate_cca_references(freqs, srate=250, T=1, n_harmonics=2)
    model = SCCA(n_components=1, n_jobs=1)
    model = model.fit(X, y, Yf)

    return model

def model_predict(X, srate=500, model=None):
    if model is None:
        raise ValueError("Model must be provided.")

    # 确保模型已经训练完成
    if not hasattr(model, 'predict'):
        raise AttributeError("The provided model does not have a predict method.")
  
    if isinstance(X, list):
        reshaped_data = []
        for array in X:
            X_ = array
            X_reshaped = np.reshape(X_, (-1, X_.shape[-2], X_.shape[-1])) 
            reshaped_data.append(X_reshaped)
        X = np.concatenate(reshaped_data)
        print("X shape after reshape:", X.shape)
        try:
            X = resample(X, up=250, down=srate)
            # Apply notch filter
            X = mne.filter.notch_filter(X, Fs=srate, freqs=50, picks=np.arange(X.shape[1]), method='fir', fir_window='hamming',
                                fir_design='firwin', verbose=False)
            # Apply bandpass filter
            X = mne.filter.filter_data(X, sfreq=srate, l_freq=8, h_freq=55, l_trans_bandwidth=2, h_trans_bandwidth=5,
                               method='fir', phase='zero-double', verbose=False)
            X = X - np.mean(X, axis=-1, keepdims=True)
            X = X / np.std(X, axis=(-1, -2), keepdims=True)
            print('X after resample:',X.shape)
            
            X_arrays = []

            # 使用循环将数组拆分成形状为 (1, 8, 250) 的子数组，并存储到列表中
            for i in range(X.shape[0]):
                split_array = X[i:i+1, :, :]
                X_arrays.append(split_array)
            print('X_arrays',len(X_arrays))
            p_labels = model.predict(X_arrays)
            return p_labels
        except Exception as e:
            print('model predict error',e)
            
    else:
        print("X shape before reshape:", X.shape)
        X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
        X = resample(X, up=250, down=srate)
        X = X - np.mean(X, axis=-1, keepdims=True)
        X = X / np.std(X, axis=(-1, -2), keepdims=True)
        print("X shape after reshape:", X.shape)
        
        p_labels = model.predict(X)
        return p_labels

def offline_validation(X, y, srate=500):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
        acc = np.mean(kfold_accs)
    return acc

class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        super().__init__(timeout=timeout, name=worker_name)
        self.label_queue = []

    def pre(self):
        #读取离线数据
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        print("Loding train data successfully")
        # Compute offline acc
        acc = offline_validation(X, y, srate=self.srate)
        print("Current Model accuracy:{:.2f}".format(acc))
        #训练个体模型
        self.estimator = train_model(X, y, srate=self.srate)
        self.ch_ind = ch_ind

    def consume(self, data):
        #data是放大器线程传入的所有设备单试次数据的组合
        print('传入数据:',len(data))
        data_buffer = []
        try:
            if data:
                for i in range(len(data)):
                    data_ = np.array(data[i], dtype=np.float64).T
                    data_buffer.append(data_)
            
        except Exception as e:
            print('consume error',e)

        print('step1')
        data_without_last_row = [array[:-1, :] for array in data_buffer]
        print('data_without_last_row:',len(data))

        print('step2')
        #使用离线模型对待测试次分类
        p_labels = model_predict(data_without_last_row, srate=self.srate, model=self.estimator)

        print('step3')
        if p_labels is not None:
            p_labels = int(p_labels)
            self.label_queue.append(p_labels)
        else:
            print("Value is None, cannot convert to integer")

    def get_label(self):
          if self.label_queue:
              label = self.label_queue.pop(0)
              print('label is',label)
              return label
          
    def post(self):
        pass


if __name__ == "__main__":
    srate = 500
    stim_interval = [0.14, 1.14]
    #由于硬件限制此处将标签13替换为21  
    stim_labels = list(range(1, 21))
    stim_labels[12]=21
    cnts = 1
    filepath = "path_to_fif"
    runs = list(range(1, cnts+1))
    run_files = ['{:s}\\{:d}.fif'.format(
        filepath, run) for run in runs]  
    pick_chs = ['TP10', 'O2', 'OZ', 'O1', 'POZ', 'PZ', 'TP9', 'FCZ']

    #workers名称
    feedback_worker_name = 'feedback_worker'
    
    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        timeout=0.05,
        worker_name=feedback_worker_name)
    
    worker.pre()

    test = DataAcquisition()
    #按照实际情况添加设备并设置参数
    test.add_device("Niantong",port_addr = 'COM12',baudrate=9600,fs=500,num_chans=8)
    test.add_device("Nianji",port_addr = 'COM16')
    test.add_device("Neuracle",device_address = ('127.0.0.1', 8712),srate=1000,num_chans=9)
    #连接设备并注册worker和marker
    test.connect_device()
    test.start_acquisition()
    test.register(feedback_worker_name,worker,stim_interval,srate,stim_labels)

    #注意念及科技设备的接收与打标功能都在同一串口进行
    brk_port = NeuraclePort(port_addr = 'COM6')
    nj_port = test.devices[1]
    nt_port = Niantong_port(port_addr = 'COM10')

    #将每个设备的打标串口添加到列表
    port_ = []
    port_.append(brk_port)
    port_.append(nt_port)
    port_.append(nj_port)

    print('are you OK? Step 1')
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    
    print('are you OK? Step 2')
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()
    print('are you OK? Step 3')

    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 240  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    print('are you OK? Step 4')
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    
    basic_ssvep.config_index()
    basic_ssvep.config_response()
    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = port_                                 
    nrep = 2  # block数目
    online = True  # True                                       # 在线实验的标志
    response_fuc_ = worker.get_label                            #获取标签值函数
    ex.register_paradigm(
        "basic SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port=port_addr,
        nrep=nrep,
        pdim="ssvep",
        response_fuc = response_fuc_,
        online=online,
    )
    print('are you OK? Step 5')

    try:
        time.sleep(1)
        test.start(feedback_worker_name)
        time.sleep(1)
        thread = threading.Thread(target=(test.put_in_worker_queue),args=(feedback_worker_name,))
        thread.start()
        print('are you OK? Step 6')
        ex.run()
        
    except Exception as e:
        print(e)
    finally:
        test.stop()
        test.stop_put_in_worker_queue()
        time.sleep(1)
        
        #nt_port.port.close()
        
        thread.join(timeout=5)
        if thread.is_alive():
            thread._stop()
