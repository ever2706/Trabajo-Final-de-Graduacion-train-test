import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read, Stream
from datetime import date
from scipy import signal
from scipy.signal import hilbert
from obspy.signal.filter import bandpass
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy.core import UTCDateTime

zeros = [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]
poles = [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j]
response={'poles': poles, 'zeros': zeros, 'gain': 3355.4428, 'sensitivity':76.7}


#periodos = [(2245, 2275), (1228, 1258)] ##Penultimo tren ampliado
periodos = [(2906, 2936), (1902, 1932)]##Ultimo tren ampliado

#periodos = [(2240, 2280), (1223, 1263)] ##Penultimo tren (San José - Heredia)
#periodos = [(2901, 2941), (1897, 1937)]##Ultimo tren (Heredia - San José)
fig, ax2 = plt.subplots(4,2, figsize=(10, 7))#Importante dejarlo fuera del for, porque sino se crea una cada vez
#fig.suptitle("Heredia - San José", fontsize=16)

envelopes=[]
trenes=[]
amplitudmax=[]
altura_subplots=[]

sensor_ids = ["453015147", "453016529"]


startS=[UTCDateTime("20230621144622"), UTCDateTime("20230621150312")]

# #Start time C station
# startC=UTCDateTime("20230621144622")
# #Start time M station
# startM=UTCDateTime("20230621150312")

for i, sensor in enumerate(sensor_ids):
    st = read(f"{sensor}*miniseed")
    tren2 = st[2]

    tren2.detrend('demean')
    tren2.detrend('linear')
    tren2.simulate(paz_remove=response) # Assuming data in m/s
    tren2.write(f"{sensor}.HHZ.tren.mseed")
    
    fs = tren2.stats.sampling_rate 
    time = np.arange(0, len(tren2)) * 0.01
    tren2_filtrado= bandpass(tren2, freqmin=1, freqmax=100, df=fs, corners=4, zerophase=True)
    
    window_size = int(0.5 * fs) # every 10 seconds. This number can change for mores smoothness
    def calcular_envolvente_amplitud_absoluta(signal, window_size):
        abs_signal = np.abs(signal)
        
        envelope = np.convolve(abs_signal, np.ones(window_size)/window_size, mode='same')
        
        return envelope
    t_start = periodos[0][0] if i == 0 else periodos[1][0] # Tiempo inicial del periodo
    t_end = periodos[0][1] if i == 0 else periodos[1][1]   # Tiempo final del periodo

    # Obtén los índices correspondientes al rango de tiempo
    idx_start = int(t_start * fs)  # Convierte el tiempo a índice
    idx_end = int(t_end * fs)
    envelope = calcular_envolvente_amplitud_absoluta(tren2_filtrado[idx_start:idx_end], window_size)
    envelopes.append(envelope)
    trenes.append(tren2_filtrado[idx_start:idx_end])
    time = np.arange(0, len(tren2[idx_start:idx_end])) * 0.01
    ax2[0,i].plot(time, tren2_filtrado[idx_start:idx_end], color='blue' if i == 0 else 'black', label= f'{startS[i]}' )
    ax2[1,i].specgram(tren2_filtrado[idx_start:idx_end], Fs=fs, mode='psd', NFFT=600, noverlap=400, vmin=-140, vmax=-115, cmap='hot')
    ax2[2,i].plot(time, envelope, label='Envelope', color='blue' if i == 0 else 'black')
    ax2[3,0].plot(time, envelope, label='Envelope', color='blue' if i == 0 else 'black')
    ax2[3,0].plot(time, envelope, label='Envelope', color='blue' if i == 0 else 'black')
    ax2[0, 0].set_ylabel('Amplitude (m/s)')
    ax2[1, 0].set_ylabel('Frequency (Hz)')
    ax2[3, 0].set_xlabel('Time (s)')
    ax2[3, 1].set_xlabel('Time (s)')
    ax2[2, 0].set_ylabel('Envelope')
    ax2[3, 0].set_ylabel("Average")
    ylim = ax2[0, i].get_ylim()
    altura_subplots.append(ylim[1])
    ax2[0,i].legend(fontsize=7, loc= "upper right")
    amplitude_min = np.min(tren2_filtrado[idx_start:idx_end])
    amplitude_max = np.max(tren2_filtrado[idx_start:idx_end])
    amplitudmax.append(amplitude_max)
    print('Rango de amplitudes: {} - {}'.format(amplitude_min, amplitude_max))
im = ax2[1, 1].specgram(tren2_filtrado[idx_start:idx_end], Fs=fs, mode='psd', NFFT=600, noverlap=400, vmin=-140, vmax=-115, cmap='hot')
cax = fig.add_axes([1.01, 0.15, 0.04, 0.7])
colorbar = plt.colorbar(im[3], cax=cax, aspect=10, shrink=0.6)
colorbar.set_label('Intensity')
envelopes=np.array(envelopes)
trenes=np.array(trenes)
print(trenes.shape)
mean_envelope=np.mean(envelopes, axis=0)
print(mean_envelope.shape)
time_envelope=np.arange(0,len(mean_envelope))*0.01
ax2[3,1].plot(time_envelope, mean_envelope, color="r")

fig.tight_layout()

plt.show()

#Grafica

print(ylim)

