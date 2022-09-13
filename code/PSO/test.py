import numpy as np
import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Activation, Dense, Dropout, GRU #, ConvLSTM2D
from keras.layers import Flatten, LeakyReLU, MaxPooling3D, TimeDistributed,MaxPooling1D
from keras.layers.convolutional import Conv3D, Conv1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import os
import util

from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(42)
from keras import backend as K


def p_error(bas, step):
    bs = 64
    lr = 0.0001
    n1 = bas[0]
    n2 = bas[1]
    k1 = bas[2]
    k2 = bas[3]
    dens1 = 256


    data_dir="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/prepared_input_files/symmetric_sb_model/"
    test_dir="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/test_set/"
    test_dir_as="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/test_set_as/"

    cur_dir  = os.getcwd()


    n_features=1

    epoch=30
    test_size  = 0.2
    ntest=50

    #################################

    x,y, memory, ndata = util.read_data(data_dir)

    indices = np.arange(x.shape[0])
    split = train_test_split(x, y, indices, test_size=test_size, random_state=999)
    x_train, x_test, y_train, y_test, idx1, idx2 = split

    print (" Ntrain = %d "%(x_train.shape[0]),flush=True)
    print (" Ntest = %d "%(x_test.shape[0]),flush=True)


    print (" memory = %d "%(memory),flush=True)
    print (" nunits1 = %d "%(n1),flush=True)
    print (" nunits2 = %d "%(n2),flush=True)
    print (" kernel1 = %d "%(k1),flush=True)
    print (" kernel2 = %d "%(k2),flush=True)
    print (" ndense1 = %d "%(dens1),flush=True)
    print (" batch_size = %d "%(bs),flush=True)
    print (" epochs = %d "%(epoch),flush=True)
    print (" lr = %9.7f "%(lr),flush=True)

    #####################################################



    traj,x_val, time, ntimes =util.read_test_data(test_dir, memory, ntest)
    traj_as,x_val_as, time, ntimes =util.read_test_data(test_dir_as, memory, ntest)



    #####################################################333

    model = Sequential()
    model.add(Conv1D(n1, k1 , activation='relu' ,input_shape=(memory, n_features),data_format='channels_last' ))
    model.add(Conv1D(n2, k2 , activation='relu', data_format='channels_last'))
    model.add(MaxPooling1D(pool_size=2, data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(dens1))
    model.add(Activation('relu'))
    model.add(Dense(n_features))
    model.add(Activation('linear'))






    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=optimizers.Adam(lr=lr),
                  metrics=['mse'])

    print(model.summary())

    history = model.fit(x_train, y_train,
                        batch_size=bs,
                        epochs=epoch,
                        verbose=True,
                        validation_data=(x_test, y_test)
                        #callbacks=callbacks,
                        #validation_split=validation_size
                       )

    stri='my_model'+str(step)+'.h5'
    model.save(stri)


    util.plot_val_acc(cur_dir,history, step)

    ######## out trainig  Test files #####

    traj1=np.zeros_like(traj)
    error_t=0.0

    for i in range(ntest):
        error=0.0
        traj1[i,:,:]=traj[i,:,:]
        for n in range(ntimes-memory):
            x_inp=traj1[i,n:n+memory,:].reshape(1,memory,1)
            #print(x_inp)
            yhat = model.predict(x_inp, verbose=False)
            traj1[i,n+memory,:]=yhat[:,:]
        util.plot_pd(cur_dir, time , traj[i,:,0], traj1[i,:,0],i, step )
        util.save(cur_dir, time , traj[i,:,0], traj1[i,:,0], i, step )

        error = np.sum(np.abs(traj[i,memory:,0]- traj1[i,memory:,0]))/len(np.abs(traj[i,memory:,0]))

        error_t += error

        print (" Errors %d : %10.5f "%(i,error))


    print (" Total and average errors %10.5f %10.5f \n"%(error_t,error_t/ntest))



    ########## Asymmetric testing files ####

    traj1_as=np.zeros_like(traj_as)
    error_t_as=0.0

    for i in range(ntest):
        error_as=0.0
        traj1_as[i,:,:]=traj_as[i,:,:]
        for n in range(ntimes-memory):
            x_inp_as=traj1_as[i,n:n+memory,:].reshape(1,memory,1)
            #print(x_inp)
            yhat_as = model.predict(x_inp_as, verbose=False)
            traj1_as[i,n+memory,:]=yhat_as[:,:]
        util.plot_pd(cur_dir, time , traj_as[i,:,0], traj1_as[i,:,0],i, step + "as" )
        util.save(cur_dir, time , traj_as[i,:,0], traj1_as[i,:,0], i, step + "as" )

        error_as = np.sum(np.abs(traj_as[i,memory:,0]- traj1_as[i,memory:,0]))/len(np.abs(traj_as[i,memory:,0]))

        error_t_as += error_as

        print (" Errors %d : %10.5f "%(i,error_as))


    print (" Total and average transfer errors %10.5f %10.5f \n"%(error_t_as,error_t_as/ntest))

    del model
    K.clear_session()

    return error_t




    #file = open('Random_search.txt', "a")
    #file.write("%d %f %f %f %f %f %f %f %10.5f %10.5f \n"%
    #             (step, memory, n1, n2, dens1 , bs, epoch, lr, aerror, aerror/ntest  ))
    #file.close()
