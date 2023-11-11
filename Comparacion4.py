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

fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
#fig.suptitle("Comparison of energy from different sources for the same node", fontsize=16)
## en los argumentos al llamar la función, nombrar una variable d (distancia) para saber 
# 6como tomar star and end 


com=["HHE", "HHN", "HHZ"]
def processing_train(node_id, fechas_r,starttime, names,t1, d, t2, color, color_l):
    
    zeros = [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]
    poles = [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j]
    response={'poles': poles, 'zeros': zeros, 'gain': 3355.4428, 'sensitivity':76.7}
    #1. Leemos los datos para el nodo seleccionado. 3 componentes. 
    st = read(f'*{node_id}*{fechas_r}*miniseed', starttime=starttime-50, endtime=starttime+60)
    #2. Removemos el mean, el trend y la respuesta de instrumento. 
    


    for i, tr in enumerate(st): 
        tr.detrend('demean')
        tr.detrend('linear')
        tr.simulate(paz_remove=response)
        #tr.data *=100
        tr.filter('highpass', freq=0.1, corners=2, zerophase=True)
        time_vec = tr.times(type='utcdatetime')
        #component = tr.stats.channel
        n1 = starttime - (t2 + d) 
        n2 = starttime - t2 
        #com.append(component)         
        # seleccionando una ventana de 30 segundos de ruido antes de la llegada del tren
        noise_window_idx=np.where(np.logical_and(time_vec>=n1, time_vec <= n2))
        timenoise = time_vec[noise_window_idx] 
        datanoise = tr.data[noise_window_idx]

        # seleccionando una ventana de señal
        d1 = starttime + t1  
        d2 = starttime + (t1 + d)
        data_window_idx = np.where(np.logical_and(time_vec>=d1, time_vec <= d2))
        timedata = time_vec[data_window_idx]
        datadata = tr.data[data_window_idx]
        nw = len(timenoise)*tr.stats.delta
        #calculando la FFT del ruido
        Py1 = MTSpec(datanoise, nw=nw, kspec=7, dt=tr.stats.delta)
        freq1, spec1 = Py1.rspec()

        Py2 = MTSpec(datadata, nw=nw, kspec=7, dt=tr.stats.delta)
        freq2, spec2 = Py2.rspec()

        line1, = axes[i].loglog(freq1, spec1, color=f'{color}', ls='--', label='Noise',  alpha=0.5)
        line3,= axes[i].loglog(freq2, spec2, color=f'{color}', label=f'{names}')
        axes[i].set_ylabel("Amplitude (m/s)", fontstyle='italic')
        #axes[i].text(25, 10**-19, f'{com[i]}' if names=="Tremor Corto" else "")
        axes[i].text(25, 10**-23, f'{com[i]}' if names=="Tremor Corto" else "")
        if i==2:
           axes[i].set_xlabel("Frequency (Hz)", fontstyle='italic')
           axes[i].set_xlim(0.6, 10**1.8)
           axes[i].legend()
           
        
        

    
sensor_ids = ["453015147", "453016529"]
names=["Tren", "Tremor Corto", "Explosion", "Regional"]
fechas_r=["..0.1.2023.06.21","..0.0014.2023.07.10","..0.0022.2023.07.18"]#Ultimo es para regionales
color=["k", "blue", "green","#800080", "red"]
color_l=["grey", "lightblue", "lightgreen", "#E6E6FA", "lightpink"]

starttime1 = UTCDateTime(2023, 6, 21, 15, 23, 54, 970000)
starttime_tremor = UTCDateTime(2023, 7, 10, 14, 41, 10)
starttime_exp = UTCDateTime(2023, 7, 10, 0, 43, 54)
starttime_reg = UTCDateTime(2023, 7, 18, 22, 3, 49)

processing_train(sensor_ids[0], fechas_r[0],starttime1, names[0], 2, 8, 12, color[0], color[0])
processing_train(sensor_ids[0], fechas_r[1], starttime_tremor, names[1], 0, 16, 5, color[1], color[1])
processing_train(sensor_ids[0], fechas_r[1], starttime_exp, names[2], 0.03, 7, 3, color[3], color[3])
processing_train(sensor_ids[0], fechas_r[2], starttime_reg, names[3], 0.90, 34, 6, color[4], color[4])




plt.tight_layout()
plt.show()
print(com)