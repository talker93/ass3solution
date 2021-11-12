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
test_blocked,t = block_audio(test_aud,1024, 512, fs)
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
    print("X.shape",X.shape)
    freq=np.fft.fftfreq(xb[0].size,1/fs)
    print("freq.shape",freq.shape)
    freqs = freq[:int(xb[0].size/2)+1]
    print("freqs.shape", freqs.shape)
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(np.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        # freq=np.fft.fftfreq(xb[0].size,1/fs)
        # freqs[n]=freq[:int(xb[0].size/2)+1]
        # compute magnitude spectum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 
    return X,freqs

def compute_spectrogram_adj(xb, fs):
    numBlocks = xb.shape[0]
    blockSize = xb.shape[1]
    afWindow = compute_hann(blockSize)
    X = np.zeros(((blockSize/2+1), numBlocks))
    freq = np.fft.fftfreq(blockSize,1/fs)
    freqs = freq[:int(blockSize/2+1)]
    for n in range(0, numBlocks):
        tmp = abs(np.fft.fft(xb[n:] * afWindow))*2/blockSize
        X[:,n] = tmp[:math.ceil(blockSize/2+1)]
    return X, freqs
    
    

S, f = compute_spectrogram(test_blocked, 44100)
S2, f2 = compute_spectrogram(test_blocked, 44100)


