# -*- coding: utf-8 -*-
"""
Amplifiers.

"""
import datetime
import socket
import struct
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from queue import Queue
import queue

import serial
import ctypes
import re
my_dll = ctypes.CDLL("path_to_LinkMe.dll")
my_dll.dataProtocol.argtypes = (ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int)
my_dll.dataProtocol.restype = ctypes.c_int
my_dll.getElectricityValue.restype = ctypes.c_int
my_dll.getFallFlag.argtypes = (ctypes.POINTER(ctypes.c_int), )
my_dll.getData.restype =ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
my_dll.getDataCurrIndex.argtypes = (ctypes.POINTER(ctypes.c_long), )

from .logger import get_logger
from .workers import ProcessWorker

logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")


class RingBuffer(deque):
    """Online data RingBuffer.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        None
    Parameters
    ----------
        size: int,
            Size of the RingBuffer.
    """

    def __init__(self, size=1024):
        """Ring buffer object based on python deque data
        structure to store data.

        Parameters
        ----------
        size : int, optional
            maximum buffer size, by default 1024
        """
        super(RingBuffer, self).__init__(maxlen=size)
        self.max_size = size

    def isfull(self):
        """Whether current buffer is full or not.

        Returns
        ----------
        boolean
        """
        return len(self) == self.max_size

    def get_all(self):
        """Access all current buffer value.

        Returns
        ----------
        list
            the list of current buffer
        """
        return list(self)


class Marker(RingBuffer):
    """Intercept online data.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
        interval: list,
            Time Window.
        srate: int,
            Amplifier setting sample rate.
        events: list,
            Event label.
    """

    def __init__(
        self, interval: list, srate: float, events: Optional[List[int]] = None
    ):
        self.events = events
        if events is not None:
            self.interval = [int(i * srate) for i in interval]
            self.latency = 0 if self.interval[1] <= 0 else self.interval[1]
            max_tlim = max(0, self.interval[0], self.interval[1])
            min_tlim = min(0, self.interval[0], self.interval[1])
            size = max_tlim - min_tlim
            if min_tlim >= 0:
                self.epoch_ind = [self.interval[0], self.interval[1]]
            else:
                self.epoch_ind = [
                    self.interval[0] - min_tlim,
                    self.interval[1] - min_tlim,
                ]
        else:
            # continous mode
            self.interval = [int(i * srate) for i in interval]
            self.latency = self.interval[1] - self.interval[0]
            size = self.latency
            self.epoch_ind = [0, size]

        self.countdowns: Dict[str, int] = {}
        self.is_rising = True
        super().__init__(size=size)

    def __call__(self, event: int):
        """Record label position.
        Parameters
        ----------
            event: int,
                Online real-time data tagging.
        """
        # add new countdown items
        if self.events is not None:
            event = int(event)
            if event != 0 and self.is_rising:
                if event in self.events:
                    # new_key = hashlib.md5(''.join(
                    # [str(event), str(datetime.datetime.now())])
                    # .encode()).hexdigest()
                    new_key = "".join(
                        [
                            str(event),
                            datetime.datetime.now().strftime("%Y-%m-%d \
                                -%H-%M-%S"),
                        ]
                    )
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info("find new event {}".format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            if "fixed" not in self.countdowns:
                self.countdowns["fixed"] = self.latency

        drop_items = []
        # update countdowns
        for key, value in self.countdowns.items():
            value = value - 1
            if value == 0:
                drop_items.append(key)
                logger_marker.info("trigger epoch for event {}".format(key))
            self.countdowns[key] = value

        for key in drop_items:
            del self.countdowns[key]
        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        """Fetch data from buffer."""
        data = super().get_all()
        return data[self.epoch_ind[0]: self.epoch_ind[1]]


class BaseAmplifier:
    """Base Ampifier class.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    -modified log:
        2024-08-04
    """

    def __init__(self):
        self._markers = {}
        self._exit = threading.Event()
        self.detected_data = Queue()

    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def start_acquisition(self):
        pass

    @abstractmethod
    def close_connection(self):
        pass
    
    @abstractmethod
    def recv(self):
        """the minimal recv data function, usually a package."""
        pass

    def _inner_loop(self,name):
        """Inner loop in the threading."""
        self.markers[name].clear()
        self._exit.clear()
        logger_amp.info("enter the inner loop")
        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples,name)
            except Exception as e:
                print(e)
        logger_amp.info("exit the inner loop")

    def _detect_event(self, samples, name):
        """detect event label"""
        marker = self._markers[name]
        for sample in samples:
            marker.append(sample)
            if marker(sample[-1]):
                self.detected_data.put(marker.get_epoch())  
                print('detect ok {}'.format(str(self)))

class Neuracle(BaseAmplifier):
    """ An amplifier implementation for neuracle devices.
    -author: Jie Mei
    -Created on: 2022-12-04
    -modified log:
        2024-08-04
    -

    Brief introduction:
    This class is a class for get package data from Neuracle device. To use
    this class, you must start the Neusen W software first, and then click
    the DataService icon on the right part and set parameter. The default
    port is 8712, and you do not need to modifiy it.
    (warning, this class was developed under Newsen W 2.0.1 version, we are
    not sure if it supports the newer version. You could ask for support
    from the Neuracle company.)

    Args:
        device_address: (ip, port)
        srate: sample rate of device, the default value of Neuracle is 1000
        num_chans: channel of data, for Neuracle, including data
                    channel and trigger channel
    """

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 8712),
                 srate=1000,
                 num_chans=9):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._update_time = 0.04
        self.pkg_size = int(self._update_time*4*self.num_chans*self.srate)

    def set_timeout(self, timeout):
        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def recv(self):
        # wait for the socket available
        data = None
        # rs, _, _ = select.select([self.tcp_link], [], [], 9)
        try:
            raw_data = self.tcp_link.recv(self.pkg_size)
        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            data, evt = self._unpack_data(raw_data)
            data = data.reshape(len(data)//self.num_chans, self.num_chans)
        return data.tolist()

    def _unpack_data(self, raw):
        len_raw = len(raw)
        event, hex_data = [], []
        # unpack hex_data in row
        hex_data = raw[:len_raw - np.mod(len_raw, 4*self.num_chans)]
        n_item = int(len(hex_data)/4/self.num_chans)
        format_str = '<' + (str(self.num_chans) + 'f') * n_item
        unpack_data = struct.unpack(format_str, hex_data)

        return np.asarray(unpack_data), event

    def connect(self):
        self.tcp_link.connect(self.device_address)

    def start_acquisition(self):
        time.sleep(1e-2)

    def stop_transmission(self):
        self._exit.set()

    def close_connection(self):
        if self.tcp_link:
            self.tcp_link.close()
            self.tcp_link = None

class Niantong(BaseAmplifier):
    """An amplifier implementation for Niantong devices.
    -author: Xixian Lin
    -Created on: 2024-08-04
    """
    _byts = 3
    _start = 2
    _checksum = -4
    _trigger = -3
    _battery = -2
    _seq = -1
    _threshold_ratio = 0.01
    
    cmd = {
        500: b"\x55\x66\x52\x41\x54\x45\x01\x0a",
        1000: b"\x55\x66\x52\x41\x54\x45\x02\x0a",
        2000: b"\x55\x66\x52\x41\x54\x45\x03\x0a",
        4000: b"\x55\x66\x52\x41\x54\x45\x04\x0a",
        8000: b"\x55\x66\x52\x41\x54\x45\x05\x0a",
        "W": b"\x55\x66\x4d\x4f\x44\x45\x57\x0a",
        "Z": b"\x55\x66\x4d\x4f\x44\x45\x5a\x0a",
        "R": b"\x55\x66\x4d\x4f\x44\x45\x52\x0a",
        "B": b"\x55\x66\x42\x41\x54\x54\x42\x0a",
        "close": b"\x55\x66\x44\x49\x53\x43\x01\x0a",
    }
    
    def __init__(self,
                 port_addr,baudrate=9600,fs=500,num_chans=8):   
        self.port_addr = port_addr
        self.baudrate = baudrate
        self.command_wait = 0.05
        self.fs = fs
        self.num_chans = num_chans
        length_ = self.num_chans * self._byts + abs(self._checksum)
        self._threshold = int((self._start + length_) * self.fs * self._threshold_ratio)
        
        self.__buffer = bytearray()
        self.__pattern = re.compile(b"\xbb\xaa.{%d}" % length_, flags=re.DOTALL)
        self.__last_num = 255
        self.ch_idx = list(range(num_chans))[:]
        self._ratio = 0.02235174
        self._drop_count = 0
        
        super().__init__()
        
    def connect(self):
        self.port = serial.Serial(timeout=5, port=self.port_addr)
        if self.port.isOpen():
            print("串口 %s 打开成功！" %self.port_addr)
            print(self.port.name)
        else:
            print("串口 %s 打开失败！" %self.port_addr)
        try: 
            self.port.write(self.cmd[self.fs])
            time.sleep(self.command_wait)
            self.port.read_all()
        except Exception as e:
            print(e)
    
    def close_connection(self):
        self.port.write(self.cmd["R"])
        time.sleep(self.command_wait)
        self.port.read_all()
        self.port.close();
        if self.port.isOpen():  
            print("串口未关闭。")
        else:
            print("串口已关闭。")
            
    def start_acquisition(self):
        ack = self.port.write(self.cmd["W"])
        self.port.read(ack)
            
    def recv(self):
        data = None
        try:
            data_ = self.port.read(self._threshold)
        except Exception as e:
           print("读取串口数据出错:", e)
        try:
            if data_:
                data = self._unpack_data(data_)
        except Exception as e:
            print("解包出错:", e)
        return data
    
    def stop_transmission(self):
        self._exit.set()
             
    def _unpack_data(self,q):
        self.__buffer.extend(q)
        if len(self.__buffer) < self._threshold:
            return
        frames = []
        for frame_obj in self.__pattern.finditer(self.__buffer):
            frame = memoryview(frame_obj.group())
            raw = frame[self._start : self._checksum]
            if frame[self._checksum] != (~sum(raw)) & 0xFF:
                self._drop_count += 1
                err = f"|Checksum invalid, packet dropped{datetime.datetime.now()}\n|Current:{frame.hex()}"
                print(err)
                continue
            cur_num = frame[self._seq]
            if cur_num != ((self.__last_num + 1) % 256):
                self._drop_count += 1
                err = f">>>> Pkt Los Cur:{cur_num} Last valid:{self.__last_num} buf len:{len(self.__buffer)} dropped times:{self._drop_count} {datetime.datetime.now()}<<<<\n"
                print(err)
            self.__last_num = cur_num
            data = [
                int.from_bytes(
                    raw[i * self._byts : (i + 1) * self._byts],
                    signed=True,
                    byteorder="big",
                )
                * self._ratio / 1e6
                for i in self.ch_idx
            ]
            data.append(frame[self._trigger])
            frames.append(data)
        if frames:
            del self.__buffer[: frame_obj.end()]
            self.batt_val = frame[self._battery]
            return frames

class Nianji(BaseAmplifier):
    """An amplifier implementation for Nianji devices.
    -author: Xixian Lin
    -Created on: 2024-08-04
    """
    def __init__(self,
                 port_addr,baudrate=460800):
        self.port_addr = port_addr
        self.baudrate = baudrate
       
        self.data_buffer = Queue()
        self.each_length = 10 * 136
        
        super().__init__()
        
    def connect(self):
        self.port = serial.Serial(port=self.port_addr, baudrate=self.baudrate)
        if self.port.isOpen():
            print("串口 %s 打开成功！" %self.port_addr)
            print(self.port.name)
        else:
            print("串口 %s 打开失败！" %self.port_addr)
            
    def recv(self):
        try:
            com_input = self.port.read(25 * 136)
        except Exception as e:
            print("读取串口数据出错:", e)
        else:
            self.data_buffer.put(com_input)
            
        if not self.data_buffer.empty():
            data = self.data_buffer.get()#[:self.each_length]
            
        data_array = (ctypes.c_ubyte * len(data))(*data)
        dataSize  = my_dll.dataProtocol(data_array, len(data))
        
        eegData = my_dll.getData()
        data_value = [[eegData[i][j] for j in range(9)] for i in range(dataSize )]#生成二维列表
        
        return data_value
         
    def close_connection(self):
        self.port.close();
        if self.port.isOpen(): 
            print("串口未关闭。")
        else:
            print("串口已关闭。")
            
    def setData(self, label):
        if str(label) != '0':
            head_string = '4E4A3C'
            hex_label = format(label, '02X')
            label_length = len(hex_label) // 2 
            length_byte = format(label_length, '02X')
            #length_byte = '01'
            label_type = '01'
            
            send_string = head_string+length_byte+label_type+hex_label
            #xor_result = label_length ^ int(label_type,16)
            xor_result = int(length_byte,16) ^ int(label_type,16)
            
            for i in range(label_length):
                byte_value = int(hex_label[i*2:i*2+2],16)
                xor_result ^= byte_value
                
            xor_byte = format(xor_result, '02X')
            
            tail = '0D0A'
            send_string += xor_byte + tail
            
            send_string_byte = [int(send_string[i:i+2],16) for i in range(0,len(send_string),2)]
            #print(send_string)
            
            self.port.write(send_string_byte)    
        
    def start_acquisition(self):
        time.sleep(1e-2)
        
    def stop_transmission(self):
        self._exit.set()

class DeviceManager:
    """Create amplifier.
    -author: Xixian Lin
    -Created on: 2024-08-04
    """
    amplifiers = {
        "Niantong": Niantong,
        "Nianji": Nianji,
        "Neuracle":Neuracle
    }

    def create_amplifier(device_type, **kwargs):
        amplifier_class = DeviceManager.amplifiers.get(device_type)
        if amplifier_class:
            return amplifier_class(**kwargs)
        else:
            raise ValueError("Unsupported device type")

class DataAcquisition:
    """Multi-device acquisition platform.
    -author: Xixian Lin
    -Created on: 2024-08-04
    """
    def __init__(self):
        self.devices = []
        self._workers = {}
        self.exit_ = threading.Event() 
        self.data_test=[]
        self.exit_.clear()
        
    def add_device(self, device_type, **kwargs):
         amplifier = DeviceManager.create_amplifier(device_type, **kwargs)
         self.devices.append(amplifier)
         
    def connect_device(self):
        for device in self.devices:
            device.connect()
    def close_con(self):
        for device in self.devices:
            device.stop_transmission()
            device.close_connection()
   
    def clear(self):
        logger_amp.info("clear all workers")
        worker_names = list(self._workers.keys())
        for name in worker_names:
            for device in self.devices:
                device.markers[name].clear()
            #self._markers[name].clear()
            self.down_worker(name)
            self.unregister_worker(name)
        
    def start_acquisition(self):
        threads1 = []
        for device in self.devices:
            thread = threading.Thread(target=device.start_acquisition)
            threads1.append(thread)
            thread.start()
        for thread in threads1:
            thread.join()

    def register(self, name, worker:ProcessWorker, interval, srate, events):
        logger_amp.info("register worker-{}".format(name))
        self._workers[name] = worker
        
        for device in self.devices:
            try:
               marker = Marker(interval, srate, events)
               device.markers[name] = marker
            except Exception:
                print('register error')
            
    def unregister_worker(self, name: str):
        logger_amp.info("unregister worker-{}".format(name))
        del self._workers[name]    
        for device in self.devices:
            del device.markers[name]

    def up_worker(self, name):
       logger_amp.info("up worker-{}".format(name))
       self._workers[name].start()
    
    def down_worker(self, name):
        logger_amp.info("down worker-{}".format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()
     
    def start(self,name):
        """start the loop."""
        self.threads3 = []
        logger_amp.info("start the loop")
        for device in self.devices:
            thread = threading.Thread(target=device._inner_loop,args=(name,))
            self.threads3.append(thread)
        for thread in self.threads3:
            thread.start()
        print('start ok')

    def stop(self):
        """stop the loop."""
        logger_amp.info("stop the loop")
        for device in self.devices:
            device.stop_transmission
            device.close_connection
        
        logger_amp.info("waiting the child thread exit")
        for thread in self.threads3:
            thread.join()
        self.clear()
        
    def stop_put_in_worker_queue(self):
        self.exit_.set()
        
    def put_in_worker_queue(self,name):
        worker = self._workers[name]
        time.sleep(1)
        print('enter put in worker queue')
        while not self.exit_.is_set(): 
            try:      
                all_samples = []
                for device in self.devices:
                    try:
                        data = device.detected_data.get(timeout=10)
                    except queue.Empty:
                        print(f"No data available from device '{str(device)}' after 10 seconds. Skipping this device.")
                        continue
                    if data: 
                        print('length data:',len(data))
                        all_samples.append(data)
                    else:
                        break  
            except Exception as e:    
                print("Exception in put in worker queue:", e)
            try:
                if all_samples:
                    worker.consume(all_samples)
                    print('sample length is',len(all_samples))
                    all_samples.clear()
                else:
                     print('Error: No data available from any device queue.')  
            except Exception as e:    
                print("Exception in put in worker queue2:", e) 
                
        print('exit put in worker queue')    
    
