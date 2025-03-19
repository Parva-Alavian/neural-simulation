#!/bin/bash
#SBATCH -J secondsimulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --time=30:10:00

module load 2023
cd $HOME/neural-simulation
# pip install -r requirements.txt
python oscillationexperiment.py --params_set "Disconnected_abh.json" --sample_size 15 --I_1 0.2 --I_2 0.8 --I_n 31 --simulation_time 20 --dt 0.001 --experiment_name "correctsst_vip" --save_path ./freq_curves/disconnected_abh/
