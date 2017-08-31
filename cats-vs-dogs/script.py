""" utils and vgg16 are utility functions written by Jeremy Howard """
from utils import *
from vgg16 import Vgg16

""" define paths to datasets """
DATA_HOME_DIR = 'data/redux'
path = DATA_HOME_DIR + '/sample/'
test_path = DATA_HOME_DIR + '/test/' # We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

""" create model based on VGG16 architecture """
limit_mem() # clean up memory used by GPU to avoid overload
vgg = Vgg16()

""" define minibatch size and number of epochs """
batch_size=32 # multiples of 2. as high as you can make it, without running into memory issues
no_of_epochs=1

""" goal is to finetune model (a.k.a. transfer learning) """
""" we remove the last fully-connected layer and replace by a new one """
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size)
vgg.finetune(batches)
vgg.model.optimizer.lr = 0.01 # empirically adjust learning rate to avoid over and underfitting

""" finetune model """
latest_weights_filename = None
for epoch in range(no_of_epochs):
    print "Running epoch: %d" % epoch
    vgg.fit(batches, val_batches, nb_epoch=1) # notice we are passing validation batch to estimate accuracy during training
    latest_weights_filename = 'ft%d.h5' % epoch # saving weights to file
    vgg.model.save_weights(results_path+latest_weights_filename)
print "Completed %s fit operations" % no_of_epochs

""" make predictions on test dataset and save to files """
batches, preds = vgg.test(test_path, batch_size = batch_size)
filenames = batches.filenames
save_array(results_path + 'test_preds.dat', preds)
save_array(results_path + 'filenames.dat', filenames)

""" format predictions to submit to Kaggle """
isdog = preds[:,1]
isdog = isdog.clip(min=0.05, max=0.95) # we have binary predictions, but Kaggle expects probabilities
ids = np.array([int(f[8:f.find('.')]) for f in filenames])
subm = np.stack([ids,isdog], axis=1)
submission_file_name = 'submission.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')
