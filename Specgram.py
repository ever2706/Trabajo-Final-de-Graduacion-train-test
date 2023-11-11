import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read, Stream
from datetime import date
from scipy import signal
from scipy.signal import hilbert
from obspy.signal.filter import bandpass
from obspy.core import UTCDateTime
import matplotlib.dates as mdates
from multitaper import MTSpec, MTCross
import multitaper.utils as mutils

zeros = [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]
poles = [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j]
response={'poles': poles, 'zeros': zeros, 'gain': 3355.4428, 'sensitivity':76.7}

start = UTCDateTime(2023, 6, 21, 15, 23, 54, 970000) #Para San Jose Heredia Malla

sensor_ids = ["453015147", "453016529"]
fig, ax3 = plt.subplots(2,1, figsize=(16, 7), sharex=True)
plt.subplots_adjust(hspace=0.2)
#cambio para Ever#
# fig.text(0.5, 0.91, "(a)", fontsize=18, fontweight='bold')
# fig.text(0.5, 0.49, "(b)", fontsize=18, fontweight='bold')
##
# fig.text(0.09, 0.90, "a)", fontsize=15, fontweight='bold')
# fig.text(0.09, 0.45, "b)", fontsize=15, fontweight='bold')
#fig.suptitle("Frecuencia de una señal generada por un tren en movimiento (San José - Heredia) respecto al tiempo", fontsize=16)
for i, sensor in enumerate(sensor_ids):
    st = read(f"{sensor}*miniseed", starttime= start -20, endtime=start +30)
    tren2 = st[2]
    
    fs = tren2.stats.sampling_rate
    time=tren2.times(type="utcdatetime")

    Ntime = []
    for t in time:
        Ntime.append(t.datetime) 
    
    Ntime = np.array(Ntime)
    tren2.detrend('demean')
    tren2.detrend('linear')
    tren2.simulate(paz_remove=response) # Assuming data in m/s
    #tren2.data *=100
    time_energia = np.arange(0, len(tren2)) * 0.01
  
    freq_min= [0.5,  0.5, 25, 15, 10]
    freq_max=[45 , 15, 35, 25, 25]
    amplitud_maxi_freq=[]
    tiempo_ampli_freq=[]  
    ax3[i].specgram(tren2, Fs=fs, mode='psd', NFFT=600, noverlap=400, vmin=-140, vmax=-115, cmap='hot') #Sin cambios, antes de multiplicar por =100 
    ax3[1].set_xlabel("Time (s)", fontsize=18)
    ax3[i].set_ylabel("Frequency (Hz)", fontsize=18)
    ax3[i].tick_params(axis='both', labelsize=18)
plt.savefig('SoloSpecgram_(SJ-H)_dr.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()
