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



def run(t_end, changes = {}, *, dt=0.001, path:Path=None, params_set = "abh_values.json"):
    #print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t_start = time.time()
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    if params_set == None:
        params = ParameterSet("smallcirtuit.json")
    else:
        params = ParameterSet(params_set)
        print("Params set to", params_set)
    params.batch_update(changes)
    #params.J.print_matrix()
    #params.J_ampa.print_matrix()
    #print(json.dumps(params.__flat_json__(ignore_zeros=True), indent=2))
    if path is not None:
        params.save(path / 'params.json')
        params.saveDelta(path / 'params_delta.json',base_file='structure.json')
        params.saveDeltaHtml(path / 'params_delta.html',base_file='structure.json')
        params.saveHtml(path / 'params.html',keys = [])

    def calc_g_static():
      sigma = y0.serialize_g(params)
      g_vector = sigma * params.constants.tau_y
      g_matrix = np.diag(g_vector)
      g_matrix = g_matrix[:,~np.all(g_matrix == 0, axis=0)]
      g_matrix = g_matrix * 1.0
      return g_matrix
    
    g_matrix = calc_g_static()

    def model_f(y, t):
        Y = Model().deserialize(y)
        delta = Y.calcDelta(t, params)
        dy = delta.serialize()
        return dy

    def model_g(y, t):
        # Y = MyState().deserialize(y)
        # sigma = Y.serialize_g(params)
        # tau_y = params.constants.tau_y
        # g = sigma * tau_y
        # return np.diag(g_vector)
        return g_matrix.copy()

    # gen = np.random.Generator(np.random.PCG64(123))
    gen = None
    # res = sdeint.itoint(model_f, model_g, y0.serialize(), t, gen)
    res = itoint(model_f, model_g, y0.serialize(), t, gen)
    def toState(y): return Model().deserialize(y)
    t_end = time.time()
    #print(f'elapsed time: {t_end - t_start} seconds')
    return t, list(map(toState, res))

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
        
        #function to extract a plot for closing the loop

def fq_curve(I,circuit_params,params_set="ExtendedV1.json",sample_size=30,save_dir=None):
    avg_y=[]
    avg_x=[]
    avg_p=[]
    std_p=[]
    pows=[]
    for i in I:    
        experiment = {"exc1.I_back.dc": i, "exc2.I_back.dc": i}
        experiment.update(circuit_params)
        print(i)
        xs =[]
        ys=[]
        pow = []    
        for j in range(sample_size):
            print(j)
            t, res = run(30, changes = experiment , dt=0.0001, path=None, params_set=params_set)
            x,y = spectrogram(t,res,smoothing = None,max_fq= 100)
            pow.append([*max_gamma_power(x,y)])
            xs.append(x)
            ys.append(y)
            
        pows.append(pow)    
        avg_y.append(sum(ys)/sample_size)
        avg_x.append(xs[0])
        avg_p.append(np.average(np.array(pow),axis=0))
        std_p.append(np.std(np.array(pow),axis=0)/sample_size)
        
    avg_x = np.array(avg_x)
    avg_y = np.array(avg_y)
    avg_p = np.array(avg_p)
    std_p = np.array(std_p)
    if save_dir:
        np.save(arr= avg_x ,file=save_dir+"/avg_x.npy")
        np.save(arr= avg_y ,file=save_dir+"/avg_y.npy")
        np.save(arr= avg_p ,file=save_dir+"/avg_p.npy")
        np.save(arr= std_p ,file=save_dir+"/std_p.npy")
        np.save(arr= I, file = save_dir+"/I.npy")
    return avg_p,std_p,avg_x,avg_y


# TODO: save the files at the right spot, save the relevant parameters as well, pass the parameters from script,  later: connect to wandb maybe?
s= 0.35
v= 0.16
experiment = {"sst1.I_back.dc":s,"sst2.I_back.dc":s,"vip1.I_back.dc":v,"vip2.I_back.dc":v, "J_ampa.vip1.exc1":0, "J_ampa.vip2.exc2":0}
I = np.linspace(0.3,0.8,26)
sample_size = 50
experiment_name = "test"
save_dir = "./freq_curves/closing_loop" + experiment_name
os.makedirs(save_dir, exist_ok=True)
avg_p,std_p,avg_x,avg_y = fq_curve(I,experiment,params_set="Disconnected.json",sample_size=sample_size,save_dir=save_dir)