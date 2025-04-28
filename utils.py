import numpy as np
import logging 
import time 
from datetime import datetime
import json
from pathlib import Path

from src import ParameterSet, Plot
from src import ModelBase as Model
from src.integral import itoint

from src.model_base import ModelBase
import matplotlib.pyplot as plt

# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(level=logging.WARN)

from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import os
import argparse

def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def max_gamma_power(xf,yf, min_fq=15, max_fq=50):
    limit = np.where((xf<=max_fq)& (xf>=min_fq))
    x_new= xf[limit]
    y_new = yf[limit]
    max_ind = np.argmax(y_new)
    max_freq = x_new[max_ind]
    max_freq_power = y_new[max_ind]
    
    power_gamma_range = sum(y_new)
    return max_freq, max_freq_power,power_gamma_range

def spectrogram( t:np.array, res: list[ModelBase],min_fq=1, max_fq=100, smoothing = None, t_start = 1, t_end=None, **kwargs):
    obj = Plot(['exc1.r'],t_start = t_start, t_end=t_end)
    t_t, traces = obj.get_traces(t, res)
    s = traces[0]
    dt = t[1]-t[0]
    N = len(s)
    yf = fft(s)
    xf = fftfreq(N, dt)[:N//2]
    limit = np.where((xf<=max_fq)& (xf>=min_fq))
    yf = yf[0:N//2]
    if smoothing =="MAF":
        try:
            window_size = kwargs["w"]  # Adjust for more or less smoothing
        except: 
            window_size = 5
            print("window_size not provided. using default value 5")
        smoothed_yf = np.convolve(2.0/N * np.abs(yf[limit]), np.ones(window_size)/window_size, mode='same')

    elif smoothing =="Gauss":
        sigma = 2  # Adjust for more or less smoothing
        smoothed_yf = gaussian_filter1d(2.0/N * np.abs(yf[limit]), sigma)

    elif smoothing=="low pass":        
        fs = 1/dt  # Sampling frequency
        cutoff_freq = 10  # Adjust this based on noise level
        smoothed_yf = lowpass_filter(2.0/N * np.abs(yf[limit]), cutoff_freq, fs)

    else:
        smoothed_yf = 2.0/N * np.abs(yf[limit])
    return xf[limit],smoothed_yf

def smooth_out(yf,smoothing,dt=None,**kwargs):
    if smoothing =="MAF":
        try:
            window_size = kwargs["w"]  # Adjust for more or less smoothing
        except: 
            window_size = 5
            print("window_size not provided. using default value 5")
        smoothed_yf = np.convolve(yf, np.ones(window_size)/window_size, mode='same')

    elif smoothing =="Gauss":
        sigma = 2  # Adjust for more or less smoothing
        smoothed_yf = gaussian_filter1d(yf, sigma)

    elif smoothing=="low pass":        
        fs = 1/dt  # Sampling frequency
        cutoff_freq = 10  # Adjust this based on noise level
        smoothed_yf = lowpass_filter(yf, cutoff_freq, fs)
    return smoothed_yf