import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pickle
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import LoadVGG


# Get one class per unique image (Powerset approach). These are entered manually
def TrainPowerSetLabels2():
    labels_list = list([[39],[41],[49],[26],[27],[40,2,48],[37],[0],[8],[11],[13],[17],[25],[32,33]])
    train_labels = np.zeros((len(labels_list)*2, 53))
    count = 0
    for index in range(len(labels_list)):
        my_list = list([0]*53)
        for element in labels_list[index]:
            my_list[element] = 1
        train_labels[index+count] = np.asarray(my_list)
        train_labels[index+count+1] = np.asarray(my_list)
        count += 1
    return(train_labels)

def TestPowerSetLabels2():
    labels_list = list([[5,28,43],[49],[16],[9],[52],[26],[27],[21,29],[20,22],[50,24,10,26,35],[7,14,28],
                        [5,7,11,13,12,19,20,26,27,32,33,35,42,49,50,52]])
    test_labels = np.zeros((len(labels_list), 53))
    for index in range(len(labels_list)):
        my_list = list([0]*53)
        for element in labels_list[index]:
            my_list[element] = 1
        test_labels[index] = np.asarray(my_list)
    return(test_labels)

# Give probability to the same occuring ranks
def TrainPowerSetLabels():
    labels_list = list([[39],[41],[49],[26],[27],[40,2,48],[37],[0],[8],[11],[13],[17],[25],[32,33]])
    train_labels = np.zeros((len(labels_list)*2, 53))
    count = 0
    for index in range(len(labels_list)):
        my_list = list([0]*53)
        for element in labels_list[index]:
            modulo = element % 13
            if(modulo == 0): # This is for the Ace, we have all the aces in clear view so there is no need to extrapolate
                #prior knowledge
                my_list[element] = 1
            else:
                if(element > 25):
                    if(element == modulo+26):
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.5, 0.5, 1, 0.75
                    else:
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.5, 0.5, 0.75, 1
                else:
                    if(element == modulo+13):
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.75, 1, 0.5, 0.5
                    else:
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 1, 0.75, 0.5, 0.5

        train_labels[index+count] = np.asarray(my_list)
        train_labels[index+count+1] = np.asarray(my_list)
        count += 1
    return(train_labels)

def TestPowerSetLabels():
    labels_list = list([[5,28,43],[49],[16],[9],[52],[26],[27],[21,29],[20,22],[50,24,10,26,35],[7,14,28],
                        [5,7,11,13,12,19,20,26,27,32,33,35,42,49,50,52]])
    test_labels = np.zeros((len(labels_list), 53))

    for index in range(len(labels_list)):
        my_list = list([0]*53)
        for element in labels_list[index]:
            modulo = element % 13
            if(modulo == 0): # This is for the Ace, we have all the aces in clear view so there is no need to extrapolate
                #prior knowledge
                my_list[element] = 1
            else:
                if(element > 25):
                    if(element == modulo+26):
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.5, 0.5, 1, 0.75
                    else:
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.5, 0.5, 0.75, 1
                else:
                    if(element == modulo+13):
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 0.75, 1, 0.5, 0.5
                    else:
                        my_list[modulo], my_list[modulo+13], my_list[modulo+26], my_list[modulo+39] = 1, 0.75, 0.5, 0.5

        test_labels[index] = np.asarray(my_list)
    return(test_labels)

def LabelsClassifier(class_number):
    power_train = TrainPowerSetLabels2()
    power_test = TestPowerSetLabels2()
    length_train = power_train.shape[0]
    length_test = power_test.shape[0]

    if(class_number == 1):
        new_train = np.zeros((length_train, 4))
        new_test = np.zeros((length_test, 4))
        for index in range(length_train):
            for index2 in range(4):
                subset_train = power_train[index,index2*13:index2*13+13]
                if(any(i == 1 for i in list(subset_train))):
                    new_train[index,index2] = 1
        for index in range(length_test):
            for index2 in range(4):
                subset_test = power_test[index, index2 * 13:index2 * 13 + 13]
                if(any(i == 1 for i in list(subset_test))):
                    new_test[index,index2] = 1

    elif(class_number == 2):
        new_train = np.zeros((length_train, 13))
        new_test = np.zeros((length_test, 13))
        for index in range(length_train):
            for index2 in range(53):
                if(power_train[index,index2] == 1):
                    new_train[index,int(index2 % 13)] = 1
        for index in range(length_test):
            for index2 in range(53):
                if(power_test[index,index2] == 1):
                    new_test[index, int(index2 % 13)] = 1

    else:
        raise ValueError('The class number input is other than 0,1,2!')

    return new_train, new_test


# We will resize the images to an equal size to input to the convolutional neural network. We could have also used padding with
# zeros and keep the original sizes.

def Get_ResizedImagesTrain(new_width, new_height, path, vgg_layers):
    num_folders = len([name for name in os.listdir(path)])
    train_set = np.empty((num_folders * 2, 19, 13, 512))
    count = 0

    with tf.Graph().as_default() as graph:  # Create a new default graph each time, otherwise the graph size exceeds the allocated 2GB.
        with tf.Session() as sess2:

            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                im = Image.open(full_path)
                im = im.resize((new_width, new_height), Image.ANTIALIAS)
                array_image = np.reshape(np.asarray(im.getdata()), (1, new_height, new_width, 3))
                array_image = array_image.astype(np.float32)

                im_rotated = im.rotate(90)
                array_image_rotated = np.reshape(np.asarray(im_rotated.getdata()), (1, new_height, new_width, 3))
                array_image_rotated = array_image_rotated.astype(np.float32)

                vgg = LoadVGG.VGG(array_image, vgg_layers)
                vgg.load()
                avg_pool5 = getattr(vgg, 'avgpool5')

                vgg = LoadVGG.VGG(array_image_rotated, vgg_layers)
                vgg.load()
                avg_pool5_rotated = getattr(vgg, 'avgpool5')

                train_set[count] = sess2.run(avg_pool5)
                train_set[count + 1] = sess2.run(avg_pool5_rotated)
                count += 2
        sess2.close()

    return(train_set)

def Get_ResizedImagesTest(new_width, new_height, path, vgg_layers):
    num_folders = len([name for name in os.listdir(path)])
    test_set = np.empty((num_folders, 19, 13, 512))
    count = 0

    with tf.Graph().as_default() as graph:  # Create a new default graph each time, otherwise the graph size exceeds the allocated 2GB.
        with tf.Session() as sess2:
            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                im = Image.open(full_path)
                im = im.resize((new_width, new_height), Image.ANTIALIAS)
                array_image = np.reshape(np.asarray(im.getdata()), (1, new_height, new_width, 3))
                array_image = array_image.astype(np.float32)

                vgg = LoadVGG.VGG(array_image, vgg_layers)
                vgg.load()
                avg_pool5 = getattr(vgg, 'avgpool5')

                test_set[count] = sess2.run(avg_pool5)
                count += 1
        sess2.close()

    return(test_set)

def GetData(new_width, new_height):
    vgg_layers = LoadVGG.getModel()
    paths = list([r'C:/Users/pinouche/Downloads/playing_cards_data__1_/data/training/',
                  r'C:/Users/pinouche/Downloads/playing_cards_data__1_/data/test_1/',
                  r'C:/Users/pinouche/Downloads/playing_cards_data__1_/data/test_2/'])

    if os.path.exists('train_set.p'):
        train_set = pickle.load(open("train_set.p", "rb"))
    else:
        train_set = Get_ResizedImagesTrain(new_width, new_height, paths[0], vgg_layers)
        pickle.dump(train_set, open("train_set.p", "wb"))

    if os.path.exists('test_set.p'):
        test_set = pickle.load(open("test_set.p", "rb"))
    else:
        test_set = Get_ResizedImagesTest(new_width, new_height, paths[1], vgg_layers)
        pickle.dump(test_set, open("test_set.p", "wb"))

    return train_set, test_set

def GetTfDatasets(new_width, new_height, batch_size, classifier_number):
    train, test = GetData(new_width, new_height)

    if (classifier_number == 0):
        train_labels = TrainPowerSetLabels()
        test_labels = TestPowerSetLabels()
    else:
        train_labels, test_labels = LabelsClassifier(classifier_number)

    train_labels = tf.convert_to_tensor(train_labels, np.float32)
    train = tf.convert_to_tensor(train, np.float32)  # Convert to tensor and put into a tf.Dataset for the train data
    train_data = tf.data.Dataset.from_tensor_slices((train, train_labels))
    train_data = train_data.shuffle(10000)
    train_data = train_data.batch(batch_size)

    test_labels = tf.convert_to_tensor(test_labels, np.float32)
    test = tf.convert_to_tensor(test, tf.float32)
    test_data = tf.data.Dataset.from_tensor_slices((test, test_labels))
    test_data = test_data.batch(batch_size)

    return train_data, test_data

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def binary_activation(x):

    tensor = tf.constant([0.5])
    cond = tf.less(x, tensor)
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return(out)

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def Combine(my_list, n_test):
    final_array = np.reshape(np.asarray(my_list[0]), (n_test, 53))
    for index in range(n_test):
        for index2 in range(4):
            if(my_list[1][index][0][index2] == 1.0):
                final_array[index,index2*13:index2*13+13] += 1

        final_array[index] += np.concatenate([np.tile(my_list[2][index][0],4),[0]])
    return(final_array)