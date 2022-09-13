import numpy as np
import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Activation, Dense, Dropout, GRU, SimpleRNN #, ConvLSTM2D
from keras.layers import Flatten, LeakyReLU, MaxPooling3D, TimeDistributed
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


from keras import optimizers
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import os
import util
from keras.models import load_model
from time import time as TM

from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(42)

tensorflow.config.threading.set_inter_op_parallelism_threads(int(os.environ.get('SLURM_NTASKS', 1)))
tensorflow.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))


data_dir="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/prepared_input_files/symmetric_sb_model/"
test_dir="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/test_set/"
test_dir_as="/work/akanane/users/Luis/Paper_ML_QD/Pavlo_data/data/test_set_as/"

#data_dir="/home/leherrera/Documents/Luis/Pavlo_data/data/prepared_input_files/symmetric_sb_model/"
#test_dir="/home/leherrera/Documents/Luis/Pavlo_data/data/test_set/"

cur_dir  = os.getcwd()



n_features=1

bs=128
epoch=30
lr=0.0001

n1= 65 #int(float(sys.argv[1])) # 
n2= 50 #int(float(sys.argv[2])) #

spec_dir=sys.argv[1]+"_"+sys.argv[2]

dens1=256
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
print (" ndense1 = %d "%(dens1),flush=True)
print (" batch_size = %d "%(bs),flush=True)
print (" epochs = %d "%(epoch),flush=True)
print (" lr = %9.7f "%(lr),flush=True)

#####################################################



traj,x_val, time, ntimes =util.read_test_data(test_dir, memory, ntest)
traj_as,x_val_as, time, ntimes =util.read_test_data(test_dir_as, memory, ntest)




#####################################################333


model = Sequential()
model.add(SimpleRNN(n1, input_shape=(memory, n_features), return_sequences=True))
model.add(SimpleRNN(n2, return_sequences=True))
model.add(Flatten())
model.add(Dense(dens1))
model.add(Activation('relu'))
model.add(Dense(n_features))
model.add(Activation('linear'))


model.compile(loss=keras.losses.mean_squared_error,
              optimizer=optimizers.Adam(lr=lr),
              metrics=['mse'])

print(model.summary())

start_T = TM()

history = model.fit(x_train, y_train,
                    batch_size=bs,
                    epochs=epoch,
                    verbose=True,
                    validation_data=(x_test, y_test)
                    #callbacks=callbacks,
                    #validation_split=validation_size
                   )

time_T=TM()-start_T
print (" Training Time  %10.5f "%(time_T))

util.plot_val_acc(cur_dir, spec_dir ,history, 0)
model.save(cur_dir+"/" +spec_dir+ "/" +'my_model.h5')

######## out trainig  Test files #####3


#y_val=np.zeros((ntest, ntimes ,1))


#total_error=0.0
#for i in range(ntest):
#    error=0.0
#    y_val[i,:,:]=x_val[i,:,:]

traj1=np.zeros_like(traj)
error_t=0.0
Total_T=0.0

for i in range(ntest):
    error=0.0
    traj1[i,:,:]=traj[i,:,:]
    start_P=TM()
    for n in range(ntimes-memory):
        x_inp=traj1[i,n:n+memory,:].reshape(1,memory,1)
        #print(x_inp)
        yhat = model.predict(x_inp, verbose=False)
        traj1[i,n+memory,:]=yhat[:,:]
    time_P=TM()-start_P
    print (" Prediction  Time %d :  %10.5f "%(i,time_P))

    util.plot_pd(cur_dir, spec_dir, time , traj[i,:,0], traj1[i,:,0],i)
    util.save(cur_dir, spec_dir, time , traj[i,:,0], traj1[i,:,0], i )

    error = np.sum(np.abs(traj[i,memory:,0]- traj1[i,memory:,0]))/len(np.abs(traj[i,memory:,0]))

    #error /= ntest
    error_t += error
    Total_T+=time_P
    print (" Errors %d : %10.5f "%(i,error))



print (" Total and average sym_errors %10.5f %10.5f \n"%(error_t,error_t/ntest))

print (" Total and average time %10.5f %10.5f \n"%(Total_T,Total_T/ntest))

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
    util.plot_pd(cur_dir, spec_dir  , time , traj_as[i,:,0], traj1_as[i,:,0],str(i) + "as" )
    util.save(cur_dir, spec_dir , time , traj_as[i,:,0], traj1_as[i,:,0], str(i)+ "as" )

    error_as = np.sum(np.abs(traj_as[i,memory:,0]- traj1_as[i,memory:,0]))/len(np.abs(traj_as[i,memory:,0]))

    error_t_as += error_as

    print (" Errors %d : %10.5f "%(i,error_as))


print (" Total and average transfer trans_errors %10.5f %10.5f \n"%(error_t_as,              error_t_as/ntest))





















#file = open('Random_search.txt', "a")
#file.write("%d %f %f %f %f %f %f %f %10.5f %10.5f \n"%
#             (step, memory, n1, n2, dens1 , bs, epoch, lr, aerror, aerror/ntest  ))
#file.close()
