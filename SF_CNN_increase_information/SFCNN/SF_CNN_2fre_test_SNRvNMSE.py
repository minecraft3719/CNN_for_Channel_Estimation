from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, normalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
session = tf.Session(config=config)
import scipy.io as sio
import time

Nt=32
Nt_beam=32
Nr=16
Nr_beam=16

# DFT matrix
def DFT_matrix(N):
    m, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    D = np.power( omega, m * n )
    return D

F_DFT=DFT_matrix(Nt)/np.sqrt(Nt)
F_RF=F_DFT[:,0:Nt_beam]
F=F_RF
F_conjtransp=np.transpose(np.conjugate(F))
FFH=np.dot(F,F_conjtransp)
invFFH=np.linalg.inv(FFH)
pinvF=np.dot(F_conjtransp,invFFH)
W_DFT=DFT_matrix(Nr)/np.sqrt(Nr)
W_RF=W_DFT[:,0:Nr_beam]
W=W_RF
W_conjtransp=np.transpose(np.conjugate(W))

scale=2
fre=2

############## testing set generation ##################
data_num_test=1000
data_num_file=1000
SNR_factor=5.9
file_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_path)
filedir = os.listdir('./2fre_data')  # type the path of testing data (different channel statistics from training data, used for performance evaluation)
count = 0

Transmit_power = [-10, -5, 0, 5, 10, 15, 20 ] #SNR derivative
NMSE_SUM = zeros(size(Transmit_power), dtype=float)

for Txpower in Transmit_power:
    print('SNR = '+ str(Txpower))
    SNR=10.0**(Txpower/10.0) # transmit power
    H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
    H_test_noisy=zeros((data_num_test,Nr_beam,Nt_beam,2*fre), dtype=float)
    n=0
    SNRr=0

    for filename in filedir:
        newname = os.path.join('./2fre_data', filename)
        data = sio.loadmat(newname)
        channel = data['ChannelData_fre']
        for i in range(data_num_file):
            for j in range(fre):
                a=channel[:,:,j,i]
                H = np.transpose(a)
                H_re = np.real(H)
                H_im = np.imag(H)
                H_test[n*data_num_file+i, :, :, 2 * j] = H_re / scale
                H_test[n*data_num_file+i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NpinvF = np.dot(N, pinvF)
                Y = H + 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
                SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
                Y_re = np.real(Y)
                Y_im = np.imag(Y)
                H_test_noisy[n*data_num_file+i, :, :, 2 * j] = Y_re / scale
                H_test_noisy[n*data_num_file+i, :, :, 2 * j + 1] = Y_im / scale
        n = n + 1
    print(n)
    print(SNRr/(data_num_test*fre))
    print(H_test.shape,H_test_noisy.shape)
    index3 = np.where(abs(H_test) > 1)
    row_num = np.unique(index3[0])
    H_test = np.delete(H_test, row_num, axis=0)
    H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)
    print(len(row_num))
    print(H_test.shape, H_test_noisy.shape)
    print(((H_test)**2).mean())


    # load model
    CNN = load_model('CNN_UMi_3path_2fre_SNRminus_10dB_200ep.hdf5')

    t1=time.time()
    decoded_channel = CNN.predict(H_test_noisy)
    t2=time.time()
    nmse2=zeros((data_num_test-len(row_num),1), dtype=float)
    for n in range(data_num_test-len(row_num)):
        MSE=((H_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_test[n,:,:,:])**2).sum()
        nmse2[n]=MSE/norm_real
    NMSE_SUM[count] = nmse2.sum()/(data_num_test-len(row_num))
    print(NMSE_SUM[count])  # calculate NMSE after training stage (testing performance)
    count = count + 1
print(Transmit_power)
print(NMSE_SUM)
#print(t2-t1)