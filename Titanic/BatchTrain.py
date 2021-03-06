import pandas as pd
import numpy as np
import tensorflow as tf
from NN import NN
import sys
import time
##TODO This really needs cleanup
def get_train_data(number_of_samples):
	df = pd.read_csv("corrected_train.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_survived = df["Survived"]
	df.drop(["Survived"], 1, inplace=True)
	n_training_samples = np.min([len(df.index), number_of_samples])
	indices = np.random.choice(range(len(df.index)), n_training_samples,
		replace = False)

	X_array = np.array([df.ix[ind].values.reshape([7]) for ind in indices])
		
	y_targets_array = np.array([df_survived.ix[ind] for ind in indices])
	
	return np.array(X_array), np.array(y_targets_array)

def get_validation_data(number_of_samples):
	df = pd.read_csv("corrected_val.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_survived = df["Survived"]
	df.drop(["Survived"], 1, inplace=True)
	n_training_samples = np.min([len(df.index), number_of_samples])
	indices = np.random.choice(range(len(df.index)), n_training_samples,
		replace = False)
	X_array = [df.ix[ind].values.reshape([7]) for ind in indices]
		
	y_targets_array = [df_survived.ix[ind] for ind in indices]
		
	return np.array(X_array), np.array(y_targets_array)


if len(sys.argv) > 1:
	try:
		number_of_batches = int(sys.argv[1])
	except:
		print("Faulty input. Using default value.")
		number_of_batches = 1
else:
	number_of_batches = 1


if len(sys.argv) > 2:
	try:
		number_of_samples = int(sys.argv[2])
	except:
		print("Faulty input. Using default value.")
		number_of_samples = 10
else:
	number_of_samples = 10


print("Performing training on {} batches of size {}.".format(number_of_batches, number_of_samples))
sess = tf.Session() ##TODO Should we close session? Google it
print("Building model.")
t1 = time.time()
aNN = NN(sess=sess)
t2 = time.time()
print("Build took {} seconds.".format(t2-t1))
print("Beginning training.")
t0 = time.time()
for j in range(number_of_batches):

	X_batch, y_targets = get_train_data(number_of_samples)
	print("Starting batch {}/{}".format(j+1, number_of_batches), end = "\r")
	loss = aNN.train(sess, X_batch, y_targets)

print("Finished batch {}/{}".format(number_of_batches, number_of_batches))
t3 = time.time()
print("Training completed. Duration: {}. Final loss: {}.".format(t3-t0, loss))
aNN.save_model(sess)



X_batch_validation, y_targets_validation = get_validation_data(number_of_samples)
print("Validating on {} samples.".format(len(y_targets_validation)))

loss_val = aNN.calculate_loss(sess, X_batch_validation, y_targets_validation)
print("Loss on validation set: {}.".format(loss_val))
accuracy = aNN.calculate_accuracy(sess, X_batch_validation, y_targets_validation)
print("Accuracy on validation set: {}".format(accuracy))





