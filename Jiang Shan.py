#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


fs=44100
test_aud = np.sin(2*np.pi*1500*np.arange(1 * fs)/fs)
test_blocked,t = block_audio(test_aud,2048, 1024, fs)
print(math.ceil(test_aud.size / 1024))


# # A. Maximum spectral peak based pitch tracker 
# 
# 
# - [5 points] Implement a function [X, fInHz] = compute_spectrogram(xb, fs) that computes the magnitude spectrum for each block of audio in xb (calculated using the reference block_audio() from previous assignments) and returns the magnitude spectrogram X (dimensions blockSize/2+1 X numBlocks) and a frequency vector fInHz (dim blockSize/2+1,) containing the central frequency of each bin. Do not use any third party spectrogram function. Note: remove the redundant part of the spectrum. Also note that you will have to apply a von-Hann window of appropriate length to the blocks before computing the fft. 
#    
# - [10 points] Implement a function: [f0, timeInSec] = track_pitch_fftmax(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on a block-wise maximum spectral peak finding approach. Note: This function should use compute_spectrogram().  
#   
# - [5 points] If the blockSize = 1024 for blocking, what is the exact time resolution of your pitch tracker? Can this be improved without changing the block-size? If yes, how? If no, why? (Use a sampling rate of 44100Hz for all calculations).  

# ## Spectrogram

# In[3]:


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


# ### Testing Spectrogram

# In[4]:


S,f = compute_spectrogram(test_blocked, 44100)


# In[5]:


S.shape


# In[6]:


f[np.argmax(S.T[0])]


# In[7]:


def plot_spectrogram(spectrogram, fs, hopSize):
    t = hopSize*np.arange(spectrogram.shape[1])/fs
    f = np.arange(0,fs/2, fs/2/spectrogram.shape[0])

    plt.figure(figsize = (15, 7))
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.pcolormesh(t, f, spectrogram, shading='auto')
    plt.show()


# In[8]:


plot_spectrogram(S,44100,1024)


# Plotting spectrum to test

# In[9]:


import plotly.express as px
px.line(y=S.T[0][:-1],x=f[:-1])


# ## At blocksize = 1024: time resolution = 0.011609977324263039 s
# Time resolution = hopSize / fs (assuming hopSize = blockSize/2, we get time resolution = blockSize/2fs)

# In[10]:


1024/44100


# In[11]:


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

_,t=block_audio(test_aud,1024,512,44100)
print(t[1]-t[0])
_,t=block_audio(test_aud,2048,1024,44100)
print(t[1]-t[0])


# ## [f0, timeInSec] = track_pitch_fftmax(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on a block-wise maximum spectral peak finding approach. Note: This function should use compute_spectrogram().

# In[12]:


def track_pitch_fftmax(x,blockSize,hopSize,fs):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    S,f=compute_spectrogram(xb,fs)
    numBlocks = S.T.shape[0]
    f0=np.zeros(numBlocks)
    for n in range(numBlocks):
        f0[n] = f[np.argmax(S.T[0])]
    return f0,t


# In[13]:


f0,t = track_pitch_fftmax(test_aud,1024,512,44100)
px.line(x=t,y=f0)


# # B. HPS (Harmonic Product Spectrum) based pitch tracker
# - [15 points] Implement a function [f0] = get_f0_from_Hps(X, fs, order) that computes the block-wise fundamental frequency f0 given the magnitude spectrogram X and the samping rate based on a HPS approach of order order.
# - [5 points] Implement a function [f0, timeInSec] = track_pitch_hps(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on  HPS based approach. Use blockSize = 1024 in compute_spectrogram(). Use order = 4 for get_f0_from_Hps()

# In[14]:


# generate a signal for test
test_aud2 = np.sin(2*np.pi*12000*np.arange(1 * fs)/fs) / 8 + np.sin(2*np.pi*6000*np.arange(1 * fs)/fs) / 4 + np.sin(2*np.pi*3000*np.arange(1 * fs)/fs) / 2 + test_aud
test_blocked2,t2 = block_audio(test_aud2,2048, 1024, fs)
print(math.ceil(test_aud.size / 1024))

S2,f2 = compute_spectrogram(test_blocked2, 44100)


# In[15]:


plot_spectrogram(S2,44100,1024)


# In[16]:


import plotly.express as px
px.line(y=S2.T[0][:-1],x=f[:-1])


# In[17]:


print("S2.shape",S2.shape);


# ## Implement a function [f0] = get_f0_from_Hps(X, fs, order) that computes the block-wise fundamental frequency f0 given the magnitude spectrogram X and the samping rate based on a HPS approach of order order.

# In[18]:


def get_f0_from_Hps(X, fs, order):

    f_min = 300
    f = np.zeros(X.shape[1])

    # get the first order HPS
    k_length = int((X.shape[0] - 1) / order)
    HPS = X[np.arange(0, k_length), :]
#     plt.subplot(order+1,1,1)
#     plt.plot(np.arange(0,k_length),HPS[:,0])
    k_min = (np.around(f_min / fs * 2 * (X.shape[0] - 1))).astype(int)

    # compute the HPS
    for j in range(1, order):
        X_d = X[::(j + 1), :]
        HPS *= X_d[np.arange(0, k_length), :]
#         plt.subplot(order+1, 1, j+1)
#         plt.plot(np.arange(0,k_length),X_d[:k_length,1])
#     plt.subplot(order+1, 1, order+1)
#     plt.plot(np.arange(0,k_length),HPS[:,0])
    
    # find the max position of bin, convert to herz
    f = np.argmax(HPS[np.arange(k_min, HPS.shape[0])], axis=0)
    f = (f+k_min) / (X.shape[0]-1) * fs / 2

    return (f)


# In[19]:


f0 = get_f0_from_Hps(S2, 44100, 4)
print("f0", f0)
print("f0.shape", f0.shape)


# ## Implement a function [f0, timeInSec] = track_pitch_hps(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on  HPS based approach. Use blockSize = 1024 in compute_spectrogram(). Use order = 4 for get_f0_from_Hps()

# In[20]:


def track_pitch_hps(x, blockSize, hopSize, fs):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    S,f=compute_spectrogram(xb,fs)
    numBlocks = S.T.shape[0]
    f0=np.zeros(numBlocks)
    f0 = get_f0_from_Hps(S, fs, order=4)
    return f0,t


# In[21]:


f0, t = track_pitch_hps(test_aud2,1024,512,44100)
print("f0",f0)


# # C. Voicing Detection
# 
# - [1 points] Take the  function [rmsDb] = extract_rms(xb) from the second assignment which takes the blocked audio as input and computes the RMS (Root Mean Square) amplitude of each block.
# - [6 points] Implement a function [mask] = create_voicing_mask(rmsDb, thresholdDb) which takes a vector of decibel values for the different blocks of audio and creates a binary mask based on the threshold parameter. Note: A binary mask in this case is a simple column vector of the same size as 'rmsDb' containing 0's and 1's only. The value of the mask at an index is 0 if the rmsDb value at that index is less than 'thresholdDb' and the value is 1 if 'rmsDb' value at that index is greater than or equal to the threshold. 
# - [6 points] Implement a function [f0Adj] = apply_voicing_mask(f0, mask)  which applies the voicing mask to the previously computed f0 so that the f0 of blocks with low energy is set to 0.

# ## Take the  function [rmsDb] = extract_rms(xb) from the second assignment which takes the blocked audio as input and computes the RMS (Root Mean Square) amplitude of each block.

# In[22]:


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


# ## [mask] = create_voicing_mask(rmsDb, thresholdDb) which takes a vector of decibel values for the different blocks of audio and creates a binary mask based on the threshold parameter.

# In[23]:


def create_voicing_mask(rmsDb,thresholdDb):
    # mask = np.zeros(rmsDb.shape[0])
    f = lambda x : 1 if x > thresholdDb else 0
    apply_threshold = np.vectorize(f)
    return(apply_threshold(rmsDb))


# ### Testing create_voicing_mask

# In[24]:


rms=np.array([100,101,200,1,3,5,50])
thresh = 30
create_voicing_mask(rms,thresh)


# ## [f0Adj] = apply_voicing_mask(f0, mask)  which applies the voicing mask to the previously computed f0 so that the f0 of blocks with low energy is set to 0.

# In[25]:


def apply_voicing_mask(f0,mask):
    return(f0*mask)


# ### Testing apply_voicing_mask

# In[26]:


randomLabel = np.random.randint(2, size=f0.shape[0])


# In[27]:


randomLabel * f0


# # D. Different evaluation metrics
# - [5 points] Implement a function [pfp] = eval_voiced_fp(estimation, annotation) that computes the percentage of false positives for your fundamental frequency estimation
# False Positive : The denominator would be the number of blocks for which annotation = 0. The numerator would be how many of these blocks were classified as voiced (with a fundamental frequency not equal to 0) is your estimation. 
# - [5 points] Implement a function [pfn] = eval_voiced_fn(estimation, annotation) that computes the percentage of false negatives for your fundamental frequency estimation
# False Negative: In this case the denominator would be number of blocks which have non-zero fundamental frequency in the annotation. The numerator would be number of blocks out of these that were detected as zero is the estimation.
# - [5 points] Now modify the eval_pitchtrack() method that you wrote in Assignment 1 to [errCentRms, pfp, pfn] = eval_pitchtrack_v2(estimation, annotation), to return all the 3 performance metrics for your fundamental frequency estimation.  Note: the errorCentRms computation might need to slightly change now considering that your estimation might also contain zeros.

# ## [5 points] Implement a function [pfp] = eval_voiced_fp(estimation, annotation) that computes the percentage of false positives for your fundamental frequency estimation
# False Positive : The denominator would be the number of blocks for which annotation = 0. The numerator would be how many of these blocks were classified as voiced (with a fundamental frequency not equal to 0) is your estimation. 

# In[28]:


def eval_voiced_fp(estimation, annotation):
    m = (annotation==0)
    denom=m.sum()
    num = ((m*estimation) > 0).sum()
    pfp = num/denom
    return pfp


# ### Testing eval_voiced_fp

# In[29]:


est = np.array([0,1,1,0])
ann = np.array([1,0,0,0])
eval_voiced_fp(est,ann)


# ## [5 points] Implement a function [pfn] = eval_voiced_fn(estimation, annotation) that computes the percentage of false negatives for your fundamental frequency estimation
# False Negative: In this case the denominator would be number of blocks which have non-zero fundamental frequency in the annotation. The numerator would be number of blocks out of these that were detected as zero is the estimation.

# In[30]:


def eval_voiced_fn(estimation, annotation):
    m = (annotation!=0)
    denom=m.sum()
    # Adding one to all elements in estimation so that only elements that are non-zero in annotation are set to zero in the calculation below:
    num = ((m*(estimation+1)) == 1).sum()  # counting number of elements that are non zero in annotation but 1 in annotation (after adding one to every element)
    pfn = num/denom
    return pfn


# ### Testing eval_voiced_fn

# In[31]:


est = np.array([0,1,1,0,0])
ann = np.array([1,1,1,0,1])
eval_voiced_fn(est,ann)
# (ann!=0)*est


# ## [5 points] Now modify the eval_pitchtrack() method that you wrote in Assignment 1 to [errCentRms, pfp, pfn] = eval_pitchtrack_v2(estimation, annotation), to return all the 3 performance metrics for your fundamental frequency estimation.  Note: the errorCentRms computation might need to slightly change now considering that your estimation might also contain zeros.

# In[32]:


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


# ### Testing eval_pitchtrack_v2

# In[33]:


est = np.array([ 0, 440,440,0,440])
ann = np.array([440, 0 ,440,0,440])
diffInCent = 100*(convert_freq2midi(est) - convert_freq2midi(ann))
eval_pitchtrack_v2(est,ann)
# (ann!=0)*est


# In[34]:


diffInCent[ann != 0]


# In[35]:


diffInCent


# In[36]:


ann


# ## E. Evaluation
# - In a separate function executeassign3(), generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec), apply your track_pitch_fftmax(), (blockSize = 1024, hopSize = 512) and plot the f0 curve. Also, plot the absolute error per block and discuss the possible causes for the deviation. Repeat for track_pitch_hps() with the same signal and parameters. Why does the HPS method fail with this signal?
# - Next use (blockSize = 2048, hopSize = 512) and repeat the above experiment (only for the max spectra method). Do you see any improvement in performance? 
# - Evaluate your track_pitch_fftmax() using the development set development set(see assignment 1) and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set.
# - Evaluate your track_pitch_hps() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set. 
# - Implement a MATLAB wrapper function [f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres) that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter. 
# - Evaluate your track_pitch() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512) over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)

# ## Evaluate your track_pitch_hps() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set.

# In[37]:


def read_pitch_txt(fileAddr):
    ground_truth_F0 = []
    with open(fileAddr, "r") as file:
        rows = file.readlines()
        for row in rows:
            row_content = row.split("     ")
            ground_truth_F0.append(float(row_content[2]))
    return ground_truth_F0


# In[38]:


def run_evaluation(complete_path_to_data_folder):
    print("                                                                ")
    print("                                                                ")
    print("------------------------Evaluation start!-----------------------")
    print("                                                                ")
    print("                                                                ")
    mapping = []
    for root, dirs, filenames in os.walk(complete_path_to_data_folder):
        for file in filenames:
            mapping.append(os.path.join(root, file))
    t = 1
    for txt_path in mapping:
        if os.path.splitext(txt_path)[1] == ".txt":
            print("------------------------Evaluating file", t, "---------------------------")
            groundTruthF0List = read_pitch_txt(txt_path)
            print(txt_path, ": success read!")
            tup = os.path.splitext(txt_path)[0]
            str = ''
            for item in tup:
                str = str + item
                str = str.replace('.f0.Corrected', '')
            wav_path = str + ".wav"
            sampleRate, audio = wavread(wav_path)
            print(wav_path, ": success read!")
            blocksF0List, blocksTimeList = track_pitch_hps(audio, 1024, 512, sampleRate)
            print("blocksF0List.shape",blocksF0List.shape)
            print("blocksTimeList.shape",blocksTimeList.shape)
            blocksF0 = np.asarray(blocksF0List)
            blocksTime = np.asarray(blocksTimeList)
            groundTruthF0 = np.asarray(groundTruthF0List)
            print("blocksF0.shape",blocksF0.shape)
            print("blocksTime.shape",blocksTime.shape)
            print("groundTruthF0.shape",groundTruthF0.shape)
            diffInCent = 100*(convert_freq2midi(blocksF0) - convert_freq2midi(groundTruthF0))
            RMS, PFP, PFN = eval_pitchtrack_v2(blocksF0, groundTruthF0)
            print("diffIncent.shape",diffInCent.shape)
            print("The RMS is: ",RMS)
            print("The PFP is: ",PFP)
            print("The PFN is: ",PFN)
            t = t + 1
    print("                                                                   ")
    print("                                                                   ")
    print("------------------------Evaluation finished!-----------------------")
    print("                                                                   ")
    print("                                                                   ")


# In[40]:


run_evaluation("trainData")


# ## Implement a MATLAB wrapper function [f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres) that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter.

# In[56]:


def comp_acf(inputVector, blsNormalized=True):
    inputVector_corr = np.zeros(inputVector.size)
    j = 0
    while j < inputVector.size:
        product_sum = 0
        product_sum = np.dot(inputVector[:len(inputVector)-j], inputVector[j:])
        inputVector_corr[j] = product_sum
        j = j + 1
    if blsNormalized == 1:
        inputVector_corr = inputVector_corr / inputVector_corr[0]
    return inputVector_corr

def get_f0_from_acf(r, fs):
    high_time = fs/50
    low_time = fs/2000
    r[int(high_time):] = 0
    r[:int(low_time)] = 0
    r_argmax = r.argmax()
    r_estimate = float(fs) / r_argmax
    return r_estimate

def track_pitch_acf(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    blocks_corr = np.zeros((xb.shape[0], xb.shape[1]))
    blocks_f0 = np.zeros(xb.shape[0])
    i = 0
    while i < xb.shape[0]:
        blocks_corr[i] = comp_acf(xb[i], True)
        blocks_f0[i] = get_f0_from_acf(blocks_corr[i], fs)
        timeInSec[i] = (i * hopSize) / fs
        # print("Block at", timeInSec[i] ,"s, F0 is:", blocks_f0[i], " Hz")
        i = i + 1
    return blocks_f0, timeInSec


# In[57]:


def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == "acf":
        f0, t = track_pitch_acf(x, 1024, 512, fs)
    elif method == "max":
        f0, t = track_pitch_fftmax(x, 1024, 512, fs)
    elif method == "hps":
        f0, t = track_pitch_hps(x, 1024, 512, fs)
    xb, t = block_audio(x,1024,512,fs)
    VRMS = extract_rms(xb)
    MASK = create_voicing_mask(VRMS, voicingThres)
    f0_masked = apply_voicing_mask(f0, MASK)
    return f0_masked, t
    


# ## Evaluate your track_pitch() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512) over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)

# In[58]:


def run_evaluation_adj(complete_path_to_data_folder, method, voicingThres):
    print("                                                                ")
    print("                                                                ")
    print("------------------------Evaluation start!-----------------------")
    print("                                                                ")
    print("                                                                ")
    mapping = []
    for root, dirs, filenames in os.walk(complete_path_to_data_folder):
        for file in filenames:
            mapping.append(os.path.join(root, file))
    t = 1
    for txt_path in mapping:
        if os.path.splitext(txt_path)[1] == ".txt":
            print("------------------------Evaluating file", t, "---------------------------")
            groundTruthF0List = read_pitch_txt(txt_path)
            print(txt_path, ": success read!")
            tup = os.path.splitext(txt_path)[0]
            str = ''
            for item in tup:
                str = str + item
                str = str.replace('.f0.Corrected', '')
            wav_path = str + ".wav"
            sampleRate, audio = wavread(wav_path)
            print(wav_path, ": success read!")
            blocksF0List, blocksTimeList = track_pitch(audio, 1024, 512, sampleRate, method, voicingThres)
            blocksF0 = np.asarray(blocksF0List)
            blocksTime = np.asarray(blocksTimeList)
            groundTruthF0 = np.asarray(groundTruthF0List)
            diffInCent = 100*(convert_freq2midi(blocksF0) - convert_freq2midi(groundTruthF0))
            RMS, PFP, PFN = eval_pitchtrack_v2(blocksF0, groundTruthF0)
            t = t + 1
    print("                                                                   ")
    print("                                                                   ")
    print("------------------------Evaluation finished!-----------------------")
    print("                                                                   ")
    print("                                                                   ")


# In[59]:


run_evaluation_adj("tranData", "max", -40)

