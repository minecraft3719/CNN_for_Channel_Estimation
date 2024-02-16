from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, normalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

Nt=32
Nt_beam=32
Nr=16
Nr_beam=16
Signal =-10
Noise = 10

limcount = 4
SNRvMSE = zeros((limcount),dtype = float)
for count in range (1,limcount):
    Signal = Signal + 10*count
    SNR=10.0**(-Signal/Noise) # transmit power equal to Signal over noise times 10
    print("SNR original =",SNR)
    # DFT matrix
    def DFT_matrix(N):
        m, n = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp( - 2 * np.pi * 1j / N )
        D = np.power( omega, m * n )
        return D

    # def AWGN(IFsig, SNR):
    #     dP = np.zeros(len(IFsig))
    #     P = 0

    #     for i in range(len(IFsig)):
    #         dP[i] = abs(IFsig[i])**2
    #         P = P + dP[i]

    #     P = P/len(IFsig)
    #     gamma = 10**(SNR/10)
    #     N0 = P/gamma
    #     n = ((N0/2)**(0.5))*np.random.standard_normal(len(IFsig))
    #     IF_n = np.zeros((len(IFsig),1))

    #     for i in range(len(IFsig)):
    #         IF_n[i,:] = IFsig[i] + n[i]

    #     return(IF_n)

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

    ############## training set generation ##################
    data_num_train=1000
    data_num_file=1000
    H_train=zeros((data_num_train,Nr,Nt,2*fre), dtype=float)
    H_train_noisy=zeros((data_num_train,Nr_beam,Nt_beam,2*fre), dtype=float)
    filedir = os.listdir('./2fre_data')  # type the path of training data
    n=0
    SNRr=0
    SNR_factor=5.9  # compensate channel power gain to approximate to 1

    for filename in filedir:
        newname = os.path.join('./2fre_data', filename)
        data = sio.loadmat(newname)
        channel = data['ChannelData_fre']
        for i in range(data_num_file):
            for j in range(fre):
                a=channel[:,:,j,i]
                H=np.transpose(a)
                H_re=np.real(H)
                H_im = np.imag(H)
                H_train[n*data_num_file+i,:,:,2*j]=H_re/scale
                H_train[n*data_num_file+i, :, :, 2*j+1] = H_im/scale

    ##########creating H noise and determined SNR 

                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NpinvF=np.dot(N,pinvF)
                noise = 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
                Y = H + noise

                SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
                print(SNRr)

                Y_re = np.real(Y)
                Y_im = np.imag(Y)
                H_train_noisy[n*data_num_file+i, :, :, 2 * j] = Y_re / scale
                H_train_noisy[n*data_num_file+i, :, :, 2 * j + 1] = Y_im / scale
        n=n+1
    print(n)
    print(SNRr/(data_num_train*fre))
    print(H_train.shape,H_train_noisy.shape)
    index1=np.where(abs(H_train)>1)
    row_num=np.unique(index1[0])
    H_train=np.delete(H_train,row_num,axis=0)
    H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)
    print(len(row_num))
    print(H_train.shape,H_train_noisy.shape)

    K=3
    input_dim=(Nr,Nt,2*fre)
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu', input_shape=input_dim))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Convolution2D(filters=2*fre, kernel_size=(K,K), padding='Same', activation='tanh'))

    # checkpoint
    filepath='CNN_UMi_3path_2fre_SNRminus10dB_200ep.hdf5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    adam=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')
    model.fit(H_train_noisy, H_train, epochs=5, batch_size=128, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)

    SNRvMSE[limcount-1] = SNR
    
print(SNRvMSE)


