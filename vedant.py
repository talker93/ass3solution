import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
import os
def ToolReadAudio(cAudioFilePath):    
    [samplerate, x] = wavread(cAudioFilePath)    
    if x.dtype == 'float32':        
        audio = x    
    else:        
        # change range to [-1,1)        
        if x.dtype == 'uint8':            
            nbits = 8        
        elif x.dtype == 'int16':            
            nbits = 16        
        elif x.dtype == 'int32':            
            nbits = 32        
        audio = x / float(2**(nbits - 1))    
        # special case of unsigned format    
    if x.dtype == 'uint8':        
        audio = audio - 1.    
    return (samplerate, audio)
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)


def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * 
np.arange(iWindowLength)))
def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    freq=np.fft.fftfreq(xb[0].size,1/fs)
    freqs = freq[:int(xb[0].size/2)+1]
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(np.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        # freq=np.fft.fftfreq(xb[0].size,1/fs)
        # freqs[n]=freq[:int(xb[0].size/2)+1]
        # compute magnitude spectum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 
    return X,freqs

def track_pitch_fftmax(x,blockSize,hopSize,fs):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    S,f=compute_spectrogram(xb,fs)
    numBlocks = S.T.shape[0]
    f0=np.zeros(numBlocks)
    for n in range(numBlocks):
        f0[n] = f[np.argmax(S.T[n])]
    return f0,t

def extract_rms(xb):
    # number of results
    numBlocks = xb.shape[0]
    # allocate memory
    vrms = np.zeros(numBlocks)
    for n in range(0, numBlocks):
        # calculate the rms
        vrms[n] = np.sqrt(np.dot(xb[n,:], xb[n,:]) / xb.shape[1])
    # convert to dB
    epsilon = 1e-5  # -100dB
    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)
    return (vrms)

def create_voicing_mask(rmsDb,thresholdDb):
    # mask = np.zeros(rmsDb.shape[0])
    f = lambda x : 1 if x > thresholdDb else 0
    return(np.array([f(x) for x in rmsDb]))

def apply_voicing_mask(f0,mask):
    return(f0*mask)

def eval_voiced_fp(estimation, annotation):
    m = (annotation==0)
    denom=m.sum()
    num = ((m*estimation) > 0).sum()
    pfp = num/denom
    return pfp

def eval_voiced_fn(estimation, annotation):
    m = (annotation!=0)
    denom=m.sum()
    # Adding one to all elements in estimation so that only elements that are non-zero in annotation are set to zero in the calculation below:
    num = ((m*(estimation+1)) == 1).sum()  # counting number of elements that are non zero in annotation but 1 in annotation (after adding one to every element)
    pfn = num/denom
    return pfn

def convert_freq2midi(fInHz, fA4InHz = 440):
    def convert_freq2midi_scalar(f, fA4InHz):
        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f/fA4InHz))
    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
       return convert_freq2midi_scalar(fInHz,fA4InHz)
    midi = np.zeros(fInHz.shape)
    for k,f in enumerate(fInHz):
        midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
    return (midi)

def eval_pitchtrack_v2(estimateInHz, groundtruthInHz):
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0
    # truncate longer vector
    if groundtruthInHz.size < estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0,groundtruthInHz.size)]
    elif estimateInHz.size < groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0,estimateInHz.size)]
    #calculating rms error
    diffInCent = 100*(convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))
    # rms = np.sqrt(np.mean(diffInCent[groundtruthInHz != 0]**2))
    rms = np.sqrt(np.mean(diffInCent**2))
    pfp = eval_voiced_fp(estimateInHz,groundtruthInHz)
    pfn = eval_voiced_fn(estimateInHz, groundtruthInHz)
    return rms,pfp,pfn


def executeassign3(blockSize=1024):
    #from scipy.io import wavfile
    #fs_wav, data_wav = wavfile.read("./data/filename.wav")
    # blockSize=1024
    hopSize=int(blockSize/2)
    fs=44100

    #Generate test signal
    f1=441
    f2=882
    t=1
    samples1=np.arange(t * 44100)
    samples2=np.arange(t * 44100)
    sig_a=np.sin(2*np.pi*f1*samples1/fs)
    sig_b=np.sin(2*np.pi*f2*samples2/fs)
    test_signal=np.concatenate([sig_a,sig_b],axis=0)
    samples=np.concatenate([samples1,samples2],axis=0)

    ground_truth = np.concatenate([np.ones(44100)*441,np.zeros(1),np.ones(44099)*882],axis=0)
    #Track pitch fftmaxa

    f0_fftmax, timeInSec_fftmax = track_pitch_fftmax(test_signal, blockSize, hopSize, fs)
    indices = timeInSec_fftmax*44100
    ground_truth = [441 if i <= 44100 else 882 for i in indices]
    # error=100*(convert_freq2midi(f0_fftmax) - convert_freq2midi(ground_truth))
    error = f0_fftmax - ground_truth
    #rms,pfp,pfn=eval_pitchtrack_v2(f0_fftmax, ground_truth)
    #plt.figure(8,12)
    plt.plot(timeInSec_fftmax,f0_fftmax)
    plt.plot(timeInSec_fftmax,error)
    plt.title("Predicted f0 and error") 
    plt.xlabel("time") 
    plt.ylabel("frequency") 
    plt.legend(['f0_predicted','errorHz'])
    plt.show()

    return f0_fftmax,timeInSec_fftmax,error

def run_evaluation (complete_path_to_data_folder):
    errorcents={}
    files=0
    errCentRms = 0
    for file_name in os.listdir(complete_path_to_data_folder):
        if file_name.endswith(".wav"):
            files = files+1
            name=file_name[:-4]
            print(name)
            #print(loc+name+'.wav')
            #print(loc+name+'.f0.Corrected.txt')
            sr,x = ToolReadAudio(complete_path_to_data_folder+name+'.wav')

            lut = np.loadtxt(complete_path_to_data_folder+name+'.f0.Corrected.txt')
            onset_seconds = lut[:,1]
            duration_seconds = lut[:,1]
            pitch_frequency = lut[:,2]
            quantized_frequency = lut[:,3]

            hopSize = np.ceil(x.shape[0]/duration_seconds.shape[0]).astype(int)
            blockSize = 2 * hopSize

            f0,ts = track_pitch_fftmax(x,blockSize,hopSize,sr)
            err,pfp,pfn = eval_pitchtrack_v2(f0,pitch_frequency)
            
            errorcents[name] = err
            errCentRms = errCentRms + (err ** 2)
    errCentRms = np.sqrt(errCentRms/files)
    return errCentRms

def comp_acf(inputVector, bIsNormalized = True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1
    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size-1, afCorr.size)]
    return (afCorr)
def get_f0_from_acf (r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])
    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)
    return (f)
def track_pitch_acf(x,blockSize,hopSize,fs):
    # get blocks
    [xb,t] = block_audio(x,blockSize,hopSize,fs)
    # init result
    f0 = np.zeros(xb.shape[0]
                  )
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n,:])
        f0[n] = get_f0_from_acf(r,fs)
    return (f0,t)


def track_pitch(x,blockSize,hopSize,fs,method,voicingThres):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    # Track Pitch
    if method == 'max':
        [f0, timeInSec] = track_pitch_fftmax(x, blockSize, hopSize, fs)
    if method == 'acf':
        [f0,timeInSec] = track_pitch_acf(x,blockSize,hopSize,fs)
    if method == 'hps':
        pass
        # [f0,timeInSec] = track_pitch_hps(x,blockSize,hopSize,fs)

    #Apply voicing mask
    rmsdB = extract_rms(xb)
    mask = create_voicing_mask(rmsdB,voicingThres)
    f0=apply_voicing_mask(f0,mask)

    return f0


if __name__ == '__main__':
    # executeassign3()
    # print(run_evaluation('/Users/vedant/Desktop/Programming/ACA-assignments/ass3solution/trainData/'))
    name='01-D_AMairena'
    complete_path_to_data_folder='/Users/vedant/Desktop/Programming/ACA-assignments/ass3solution/trainData/'
    sr,x = ToolReadAudio(complete_path_to_data_folder+name+'.wav')
    lut = np.loadtxt(complete_path_to_data_folder+name+'.f0.Corrected.txt')
    duration_seconds = lut[:,1]
    pitch_frequency = lut[:,2]
    hopSize = np.ceil(x.shape[0]/duration_seconds.shape[0]).astype(int)
    blockSize = 2 * hopSize

    f0,ts = track_pitch_fftmax(x,blockSize,hopSize,sr)
    err,pfp,pfn = eval_pitchtrack_v2(f0,pitch_frequency)

    print(err,pfp,pfn)