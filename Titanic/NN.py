import tensorflow as tf
import os
import numpy as np


class NN:

	def __init__(self, sess=None, save_path="./model/model.ckpt"):
		self.save_path = save_path if save_path.endswith(".ckpt") else save_path + ".ckpt"
		self._get_model(sess)
		self._build_loss()
		self._build_train_op()



	def _get_model(self, sess):
		self._build_model()

		if os.path.exists(self.save_path + ".meta"):
			self._restore_model(sess)
		else:
			self._init_model(sess)

	def _restore_model(self, sess):
		saver = tf.train.Saver()
		saver.restore(sess, self.save_path)
		print("Restored model from saved parameters.")

	def _init_model(self, sess):
		init = tf.global_variables_initializer()
		sess.run(init)
		print("Initialized model randomly.")


	def _build_model(self):

		self.input_layer = tf.placeholder(tf.float32, [None, 7])

		dense1 = tf.layers.dense(
			inputs=self.input_layer,
			units = 1024,
			activation = tf.nn.relu,
			name = "dense1")

		dense2 = tf.layers.dense(
			inputs = dense1,
			units = 512,
			activation = tf.nn.relu,
			name = "dense2")

		self.logits = tf.layers.dense(
			inputs = dense2,
			units = 2,
			name = "logits")


	def _build_loss(self):
		self.target_labels = tf.placeholder(tf.int32, [None], name="target_labels")
		onehot_labels = tf.one_hot(indices=self.target_labels, depth = 2,
			name = "onehot_labels")
		self.loss = tf.losses.softmax_cross_entropy(
			onehot_labels = onehot_labels, logits = self.logits
			)

	def _build_train_op(self):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

		self.train_op = optimizer.minimize(
			loss = self.loss,
			global_step = tf.train.get_global_step(),
			name = "train_op"
			)

	def train(self, sess, X, y_target):
		feed_dict = {self.input_layer : X, self.target_labels : y_target}
		loss, _ = sess.run(
			[self.loss, self.train_op], feed_dict
			)
		return loss

	def calculate_loss(self,sess, X, y_target):
		feed_dict = {self.input_layer : X, self.target_labels : y_target}
		loss = sess.run([self.loss], feed_dict)[0]
		return loss

	def calculate_accuracy(self, sess, X, y_target):
		feed_dict = {self.input_layer : X}

		probs = sess.run([self.logits], feed_dict)[0]

		predictions = [np.argmax(prob) for prob in probs]

		correct_predictions = [1 if predictions[j] == y_target[j] \
			else 0 for j in range(len(predictions))]

		accuracy = np.sum(correct_predictions)/len(predictions)
		return accuracy


	def save_model(self, sess):
		saver = tf.train.Saver()

		save_path = saver.save(sess, self.save_path)
		print("Model saved in {}.".format(save_path))

	def predict(self, sess, X):
		feed_dict = {self.input_layer : X}
		logits_array = sess.run([self.logits], feed_dict)[0]
		preds = [np.argmax(logits) for logits in logits_array]
		return preds

#Basic tests
if __name__ == "__main__":
	sess = tf.Session()
	aNN = NN(sess = sess)
