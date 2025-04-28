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
from utils import *

def run(t_end, changes = {}, params_set = {},*, dt=0.001,path:Path=None):
    #print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t_start = time.time()
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    if isinstance(params_set, dict):  
        params_data = params_set  # Already a dictionary, use it directly  
    elif isinstance(params_set, str):  
        with open(params_set, "r", encoding="cp1252") as f:  
            params_data = json.load(f)  # Load from file  
    else:  
        raise TypeError("params_set must be either a dictionary or a filename (string)")
    params = ParameterSet(params_data)
    params.batch_update(changes)
    #params.J.print_matrix()
    #params.J_ampa.print_matrix()
    #print(json.dumps(params.__flat_json__(ignore_zeros=True), indent=2))
    # param_dict =params.__flat_json__()
    # print(param_dict["exc1.I_back.dc"],param_dict["exc1.I_back.dc"],param_dict["J.exc1.exc2"])
    #print(params.getDelta(base_file= path/"param.json"))
    # if path is not None:
    #     params.save(path / 'params.json')
    #     params.saveDelta(path / 'params_delta.json',base_file='structure.json')
    #     params.saveDeltaHtml(path / 'params_delta.html',base_file='structure.json')
    #     params.saveHtml(path / 'params.html',keys = [])

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


        
        #function to extract a plot for closing the loop

def fq_curve(I,params_set,sample_size=30,simulation_time = 10, dt = 0.001,save_dir=  None):
    with open (params_set,"r", encoding="cp1252") as f:
        params_data = json.load(f)
    #avg_y=np.zeros(len(I))
    #avg_x=np.zeros(len(I))
    avg_p=np.zeros((len(I),3))
    std_p=np.zeros((len(I),3))
    pows=[]
    for ind,i in enumerate(I):    
        experiment = {"exc1.I_back.dc": i, "exc2.I_back.dc": i}
        print(i)
        xs =[]
        ys=[]
        pow = []    
        for j in range(sample_size):
            print(j)
            t, res = run(simulation_time, changes = experiment , dt=dt, path=None, params_set=params_data)
            x,y = spectrogram(t,res,smoothing = None,max_fq= 100)
            pow.append([*max_gamma_power(x,y)])
            xs.append(x)
            ys.append(y)
            
        pows.append(pow)    
        #avg_y[ind] = (sum(ys)/sample_size)
        #avg_x[ind] = (xs[0])
        avg_p[ind][:] = (np.average(np.array(pow),axis=0))
        std_p[ind][:] = (np.std(np.array(pow),axis=0)/sample_size)
        if (ind+1) %5 == 0 or ind == len(I)-1 :        
            if save_dir:
                #np.save(arr= avg_x ,file=save_dir+ str(ind) + "/avg_x.npy")
                #np.save(arr= avg_y ,file=save_dir+ str(ind) + "/avg_y.npy")
                np.save(arr= avg_p ,file=save_dir / f'{ind}avg_p.npy')
                np.save(arr= std_p ,file=save_dir / f'{ind}std_p.npy')
                np.save(arr= I, file = save_dir   / f'{ind}I.npy')
    return avg_p,std_p#,avg_x,avg_y

def parse_args():
    parser = argparse.ArgumentParser(description="Obtain peak frequency and peak power curves")
    parser.add_argument('--sample_size', type=int, default= 20, help="Number of simulations per data point")
    parser.add_argument('--I_1',type = float,  help="I = np.array(I_1,I_2,n)")
    parser.add_argument('--I_2',type = float,  help="I = np.array(I_1,I_2,n)")
    parser.add_argument('--I_n',type = int,  help="I = np.array(I_1,I_2,n)")
    parser.add_argument('--simulation_time',type = int, default = 10, help="length of each simulation")
    parser.add_argument('--dt',type = float, default = 0.001, help="time interval for simulation")
    parser.add_argument('--save_path', type=str, default="./results/", help="Path to save results")
    parser.add_argument('--experiment_name', type=str, help="name of the experiment")
    parser.add_argument('--params_set', type= str, default = "Disconnected_abh.json", help="base parameterset for the experiment")
    return parser.parse_args()

def main():
    args= parse_args()
    
    dt = datetime.now()
    folder = Path(f'results/{args.experiment_name}/{dt.strftime("%Y-%m-%d")}/{dt.strftime("%H%M%S")}')
    folder.mkdir(parents=True)
    
    # opens the based parameter set, makes the changes necessary for the circuit that is being experimented with and saves it
    with open (args.params_set,"r", encoding="cp1252") as f:
        params_data = json.load(f)
    params = ParameterSet(params_data)
    noise_off = {"exc1.sigma":0.0, "exc2.sigma":0.0, "pv.sigma":0.0, "sst1.sigma":0.0, "sst2.sigma":0.0, "vip1.sigma":0.0, "vip2.sigma":0.0}
    disconnect_sst_to_vip = {"J.vip1.sst1": 0.0, "J.vip2.sst2": 0.0}
    disconnect_E_to_sst_vip = {"J.sst1.exc1":0.0, "J.sst2.exc2":0.0, "J.vip1.exc1":0.0, "J.vip2.exc2":0.0, "J_ampa.sst1.exc1":0.0, "J_ampa.sst2.exc2":0.0, "J_ampa.vip1.exc1":0.0, "J_ampa.vip2.exc2":0.0}
    disconnect_sst_to_E = {"J.exc1.sst1":0.0, "J.exc.sst2":0.0}
    circuit_params = noise_off | disconnect_E_to_sst_vip | disconnect_sst_to_E
    params.batch_update(circuit_params)
    params.save(folder/'params.json')
    I = np.linspace(args.I_1,args.I_2,args.I_n)
    sample_size = args.sample_size
    simulation_time = args.simulation_time
    dt = args.dt
    
    avg_p,std_p = fq_curve(I,params_set=folder/"params.json",sample_size=sample_size,simulation_time = simulation_time, dt= dt, save_dir= folder)
    
# def main():
#     args = parse_args()
    
#     # TODO: save the files at the right spot, save the relevant parameters as well, pass the parameters from script,  later: connect to wandb maybe?
#     s= 0.32
#     v= 0.33
#     #experiment = {"sst1.I_back.dc":s,"sst2.I_back.dc":s,"vip1.I_back.dc":v,"vip2.I_back.dc":v, "J_ampa.vip1.exc1":0, "J_ampa.vip2.exc2":0}
#     experiment = {"J.exc1.sst1":0.0,"J.exc2.sst2":0.0, "J_ampa.vip1.exc1":0, "J_ampa.vip2.exc2":0}
#     # disconnect = {"J.vip1.sst1": 0.0, "J.vip2.sst2": 0.0}
#     # experiment =   disconnect 
#     dt = datetime.now()
#     folder = Path(f'results/{dt.strftime("%Y-%m-%d")}/{args.experiment_name}/{dt.strftime("%H%M%S")}')
#     print(str(folder))
#     #folder = Path(f'img/{exp}/{dt.strftime("%Y-%m-%d")}/highampa_theta')
#     folder.mkdir(parents=True)
#     I = np.linspace(args.I_1,args.I_2,args.I_n)
#     sample_size = args.sample_size
#     simulation_time = args.simulation_time
#     dt = args.dt
#     # experiment_name = args.experiment_name
#     # save_dir = args.save_path + experiment_name
    
#     #os.makedirs(save_dir, exist_ok=True)
    
#     avg_p,std_p = fq_curve(I,circuit_params=experiment,params_set=args.params_set,sample_size=sample_size,simulation_time = simulation_time, dt= dt, save_dir= str(folder))
    
    
if __name__=="__main__":
        main()
