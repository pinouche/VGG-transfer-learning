import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import utils
import LoadVGG



#Combine the results from our 3 classifiers

def Combine(my_list, n_test):
    final_array = np.reshape(np.asarray(my_list[0]), (n_test, 53))
    for index in range(n_test):
        for index2 in range(4):
            if(my_list[1][index][0][index2] == 1.0):
                final_array[index,index2*13:index2*13+13] += 1

        final_array[index] += np.concatenate([np.tile(my_list[2][index][0],4),[0]])
    return(final_array)


# Define the pieces for our model: Conv + ReLu layer, maxpool operation and fully connected layer. We can then just add/call
# these when we are building our model.

# Define a convolutional layer followed by ReLu
def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)

    return (tf.nn.relu(conv + biases, name=scope.name))


# Define the max pool operation where the input to be fed is a Conv layer + ReLu

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool


def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out


class ConvNet():
    def __init__(self, lr, batch_size, keep_prob, n_classes, skip_step, n_test, n_train, new_width, new_height,
                 classifier_number):
        self.lr = lr
        self.batch_size = batch_size
        self.keep_prob = tf.constant(keep_prob)
        self.n_classes = n_classes
        self.skip_step = skip_step
        self.n_test = n_test
        self.new_width = new_width
        self.new_height = new_height
        self.n_train = n_train
        self.classifier_number = classifier_number

        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.GetTfDatasets(new_width, new_height, self.batch_size, self.classifier_number)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            img, self.label = iterator.get_next()
            self.img = img
            # self.img = tf.reshape(img, shape = [-1, new_height, new_width, 3])

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

    # Define our model here: 2 conv layers, each followed by a max pooling and then a fully connected layer of dim 1024.
    # We have 32 filters/feature maps in the first convolution layer and 64 in the second.

    def inference(self):
        # conv1 = conv_relu(inputs=self.img,filters=32,k_size=11,stride=1,padding='SAME',scope_name='conv1')
        # pool1 = maxpool(conv1, 4, 4, 'VALID', 'pool1')

        # conv2 = conv_relu(inputs=pool1,filters=64,k_size=11,stride=1,padding='SAME',scope_name='conv2')
        # pool2 = maxpool(conv2, 4, 4, 'VALID', 'pool2')

        # feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        # pool2 = tf.reshape(pool2, [-1, feature_dim])

        feature_dim = self.img.shape[1] * self.img.shape[2] * self.img.shape[3]
        self.img = tf.reshape(self.img, [-1, feature_dim])
        fc = fully_connected(self.img, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(fc), self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')

    def loss(self):

        with tf.name_scope('loss'):
            l2 = 0.00001 * sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if ("biases" not in tf_var.name))
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss') + l2

    def optimize(self):

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def evalu(self):
        with tf.name_scope('predict'):
            self.preds = tf.nn.sigmoid(self.logits)
            self.binary_preds = utils.binary_activation(self.preds)
            self.bool_vec = tf.equal(self.binary_preds, self.label)
            self.correct_preds = tf.reduce_sum(tf.cast(tf.equal(self.binary_preds, self.label), tf.float32), axis=1)
            self.accuracy = self.correct_preds
            self.total_ones_label = utils.tf_count(self.label, 1)

    def build(self):

        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evalu()

    def train_one_epoch(self, sess, saver, init, epoch, step):
        sess.run(init)
        total_loss = 0
        n_batches = 0
        total_correct_preds = 0
        num_labels = 0
        try:
            while True:
                _, l, accuracy_batch = sess.run([self.opt, self.loss, self.accuracy])
                total_correct_preds += accuracy_batch[0]
                num_labels += self.n_classes

                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_playingcards' + str(self.classifier_number) + '/checkpoint', step)
        print('Training loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Training accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_labels))
        return step

    def eval_once(self, sess, init, epoch, step):
        sess.run(init)
        # self.training = False
        total_loss = 0
        total_correct_preds = 0
        num_labels = 0
        list_bool = []
        list_pred = []
        list_prob_preds = []
        try:
            while True:
                accuracy_batch, l, boole_pred, pred, prob_pred = sess.run(
                    [self.accuracy, self.loss, self.bool_vec, self.binary_preds, self.preds])
                total_correct_preds += accuracy_batch[0]
                num_labels += self.n_classes
                total_loss += l
                list_bool.append(boole_pred)
                list_pred.append(pred)
                list_prob_preds.append(prob_pred)
        except tf.errors.OutOfRangeError:
            pass

        print('Val accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_labels))
        print('Val loss at epoch {0}: {1}'.format(epoch, total_loss / self.n_test))
        return list_prob_preds

    def train(self, n_epochs, evaluation):

        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_playingcards' + str(self.classifier_number))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname('checkpoints/convnet_playingcards' + str(self.classifier_number) + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(1, n_epochs + 1):

                # If False we load the model we already trained and train it again
                if (evaluation == False):
                    step = self.train_one_epoch(sess, saver, self.train_init, epoch, step)
                    list_prob = self.eval_once(sess, self.test_init, epoch, step)

                # If True, we only load the model we trained and eval it.  We also gather the predictions.
                elif (evaluation == True):
                    if (epoch == n_epochs):
                        list_prob = self.eval_once(sess, self.test_init, epoch, step)
                        # Do it only once, since it is the same (we do not upgrade the weights). Or just
                        # say n_epoch = 1 to do it only once.

        return list_prob
        # writer.close()

# Define the hyperparameters to be fed in the model class

if __name__ == "__main__":

    new_width = 400
    new_height = 600
    n_test = 12
    n_train = 28

    batch_size = 1
    keep_prob = 0.75
    skip_step = 5
    n_epochs = 1
    lr = 0.0001

    pred_proba = []
    # if classifier_number= 0, then we have a powerlabel (one class per card = 53 classes). if classifier_number = 1, we only have labels
    # for the suits (num_classes = 4). if classifier_number = 2, we only have labels for the ranks (num_classes = 13).
    for classifier_number in range(3):
        if(classifier_number == 0):
            n_classes = 53
        elif(classifier_number == 1):
            n_classes = 4
        elif(classifier_number == 2):
            n_classes = 13

        model = ConvNet(lr, batch_size, keep_prob, n_classes, skip_step, n_test, n_train, new_width, new_height,
                        classifier_number)
        model.build()
        pred_prob = model.train(n_epochs, False)
        val_pred = model.train(1, True)
        pred_proba.append(val_pred)
        tf.reset_default_graph()

    results = Combine(pred_proba, n_test)
