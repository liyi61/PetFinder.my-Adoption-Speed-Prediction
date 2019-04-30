import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
from sklearn.metrics import cohen_kappa_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'



def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

breeds = pd.read_csv('./breed_labels.csv')
colors = pd.read_csv('./color_labels.csv')
states = pd.read_csv('./state_labels.csv')

train = pd.read_csv('./train/train.csv', engine='python')
test = pd.read_csv('./splitbp/test_split.csv', engine='python')
sub = pd.read_csv('./splitbp/sample_submission_split.csv', engine='python')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])
sentiment_dict = {}
for filename in os.listdir('./train_sentiment/'):
    with open('./train_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

for filename in os.listdir('./test_sentiment/'):
    with open('./test_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

train['lang'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

test['lang'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

all_data['lang'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
all_data['magnitude'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

####### Basic model
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'health', 'Free', 'score',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'desc_length', 'desc_words',
               'averate_word_length', 'magnitude']

train = train[[col for col in cols_to_use if col in train.columns]]
test = test[[col for col in cols_to_use if col in test.columns]]

cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'RescuerID', 'Fee', 'Age',
            'VideoAmt', 'PhotoAmt']

more_cols = []
for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 != col2 and col1 not in ['RescuerID', 'State'] and col2 not in ['RescuerID', 'State']:
            train[col1 + '_' + col2] = train[col1].astype(str) + '_' + train[col2].astype(str)
            test[col1 + '_' + col2] = test[col1].astype(str) + '_' + test[col2].astype(str)
            more_cols.append(col1 + '_' + col2)

cat_cols = cat_cols + more_cols

# time
indexer = {}
for col in cat_cols:
    # print(col)
    _, indexer[col] = pd.factorize(train[col].astype(str))

for col in cat_cols:
    # print(col)
    train[col] = indexer[col].get_indexer(train[col].astype(str))
    test[col] = indexer[col].get_indexer(test[col].astype(str))

y_train = train['AdoptionSpeed']
train = train.drop(['AdoptionSpeed'], axis=1)
train_data = train.as_matrix()
train_label0 = y_train.as_matrix()

train_label = np.zeros([train_label0.shape[0], 5])
for i in range(train_label0.shape[0]):
    if train_label0[i] == 0:
        train_label[i, 0] = 1
    elif train_label0[i] == 1:
        train_label[i, 1] = 1
    elif train_label0[i] == 2:
        train_label[i, 2] = 1
    elif train_label0[i] == 3:
        train_label[i, 3] = 1
    elif train_label0[i] == 4:
        train_label[i, 4] = 1

y_test = test['AdoptionSpeed']
test = test.drop(['AdoptionSpeed'], axis=1)
test_data = test.as_matrix()
test_label0 = y_test.as_matrix()

test_label = np.zeros([test_label0.shape[0], 5])
for i in range(test_label0.shape[0]):
    if test_label0[i] == 0:
        test_label[i, 0] = 1
    elif test_label0[i] == 1:
        test_label[i, 1] = 1
    elif test_label0[i] == 2:
        test_label[i, 2] = 1
    elif test_label0[i] == 3:
        test_label[i, 3] = 1
    elif test_label0[i] == 4:
        test_label[i, 4] = 1


num_examples_per_epoch_for_train = train_data.shape[0]
training_epochs = 70
batch_size = 1200
display_step = 1
LR = 0.0001


# Normalization
def norm(inputs, on_train):
    conv_mean, conv_var = tf.nn.moments(inputs, [0, 1, 2])

    scale = tf.Variable(tf.ones([inputs.shape[-1]]))
    shift = tf.Variable(tf.zeros([inputs.shape[-1]]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([conv_mean, conv_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(conv_mean), tf.identity(conv_var)

    mean, var = tf.cond(on_train, mean_var_with_update, lambda: (ema.average(conv_mean), ema.average(conv_var)))
    norm_out = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon)
    return norm_out

# Random Drop out nodes to avoid overfitting 
def Dropout(x, rate):
    with tf.name_scope('Dropout_layer'):
        return tf.nn.dropout(x, rate)


# Define Convolutional Layers
def add_layer(input, W_shape, b_shape, strides,  on_train,  layer_name, padding='SAME', activation_function=tf.nn.tanh):
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
        # Initialize weights
            Weight = tf.Variable(tf.truncated_normal(shape=W_shape, stddev=0.1, dtype=tf.float32))
        with tf.name_scope('Biases'):
        
            b = tf.Variable(tf.constant(0.01, shape=b_shape))
        with tf.name_scope('Conv'):
        
            conv_out = tf.nn.conv2d(input, Weight, strides, padding)
            conv_out_plus_b = tf.nn.bias_add(conv_out, b)
     
        with tf.name_scope('Activation'):
        #  activation, add non linear relationship
            x = activation_function(conv_out_plus_b)
    return x

# Ddfine input placeholder 
with tf.name_scope('Inputs'):
    x_placeholder = tf.placeholder(tf.float32, [None, 294])
    x = tf.reshape(x_placeholder, [-1, 294, 1, 1])
    y = tf.placeholder(tf.int32, [None, 5])
    on_train = tf.placeholder(tf.bool)
    rate_place = tf.placeholder(tf.float32)



# convolutioanl layer 1 
conv1 = add_layer(x, [3, 1, 1, 16], [16], [1, 1, 1, 1], on_train, 'Conv1_1')

# ResNet Convolutional Block 1 (including two 3*1 convolutional layers)
with tf.name_scope('Res_block1'):
    conv1_2 = add_layer(conv1, [3, 1, 16, 16], [16], [1, 1, 1, 1], on_train, 'Conv1_2')
    conv1_3 = add_layer(conv1_2, [3, 1, 16, 16], [16], [1, 1, 1, 1], on_train, 'Conv1_3',
                        activation_function=tf.identity)
    fuse1 = tf.nn.tanh(tf.add(conv1, conv1_3))

trans1 = add_layer(fuse1, [3, 1, 16, 64], [64], [1, 1, 1, 1], on_train, 'Tans1')

# ResNet Convolutional Block 2 (including two 3*1 convolutional layers)
with tf.name_scope('Res_block2'):
    conv2_1 = add_layer(trans1, [3, 1, 64, 64], [64], [1, 1, 1, 1], on_train, 'Conv2_1')
    conv2_2 = add_layer(conv2_1, [3, 1, 64, 64], [64], [1, 1, 1, 1], on_train, 'Conv2_2',
                        activation_function=tf.identity)
    fuse2 = tf.nn.tanh(tf.add(trans1, conv2_2))

trans2 = add_layer(fuse2, [3, 1, 64, 128], [128], [1, 1, 1, 1], on_train, 'Tans2')

# ResNet Convolutional Block 3 (including two 3*1 convolutional layers)
with tf.name_scope('Res_block3'):
    conv3_1 = add_layer(trans2, [3, 1, 128, 128], [128], [1, 1, 1, 1], on_train, 'Conv3_1')
    conv3_2 = add_layer(conv3_1, [3, 1, 128, 128], [128], [1, 1, 1, 1], on_train, 'Conv3_2',
                        activation_function=tf.identity)
    fuse3 = tf.nn.tanh(tf.add(trans2, conv3_2))

# vectorize features 37632
fuse_F_flat = tf.reshape(fuse3, [-1, 37632])

# Fully Connected 1
with tf.name_scope('FC1_layer'):
    with tf.name_scope('W_fc1'):
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[37632, 1024], stddev=0.1, dtype=tf.float32))
    with tf.name_scope('b_fc1'):
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[1024]))
    fc1 = tf.nn.tanh(tf.matmul(fuse_F_flat, W_fc1) + b_fc1)
    fc1_drop = Dropout(fc1, rate_place)
# Fully Connected 2
with tf.name_scope('FC1_layer'):
    with tf.name_scope('W_fc2'):
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 1024], stddev=0.1, dtype=tf.float32))
    with tf.name_scope('b_fc2'):
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[1024]))
    fc2 = tf.nn.tanh(tf.matmul(fc1_drop, W_fc2) + b_fc2)
    fc2_drop = Dropout(fc2, rate_place)
# Fully Connected 3
with tf.name_scope('FC1_layer'):
    with tf.name_scope('W_fc3'):
        W_fc3 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.1, dtype=tf.float32))
    with tf.name_scope('b_fc3'):
        b_fc3 = tf.Variable(tf.constant(0.01, shape=[512]))
    fc3 = tf.nn.tanh(tf.matmul(fc1_drop, W_fc3) + b_fc3)
    fc3_drop = Dropout(fc3, rate_place)


# Prediction Output Layer with output dimendion 5
with tf.name_scope('Output'):
    with tf.name_scope('W_fc4'):
        W_fc4 = tf.Variable(tf.truncated_normal(shape=[512, 5], stddev=0.1, dtype=tf.float32))
    with tf.name_scope('b_fc4'):
        b_fc4 = tf.Variable(tf.constant(0.01, shape=[5]))
    y_pred = tf.matmul(fc3_drop, W_fc4) + b_fc4

# Use Cross-entropy as loss function 
with tf.name_scope('Loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# To calculate accuracy
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

GT = tf.argmax(y, 1)
PD = tf.argmax(y_pred, 1)


init_op = tf.global_variables_initializer()

# to save model
saver = tf.train.Saver(tf.global_variables())

# Start Session，run the network
with tf.Session() as sess:
    # write model and parameters into ckpt file 
    ckpt = tf.train.get_checkpoint_state('./model')
    
    sess.run(init_op)
    # number of batches
    total_batches = int(num_examples_per_epoch_for_train / batch_size)
    print('Per batch Size:', batch_size)
    print('Train sample Count Per Epoch:', num_examples_per_epoch_for_train)
    print('Total batch Count Per Epoch:', total_batches)
    training_step = 0
    # training 
    for epoch in range(training_epochs):
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            # training batch
            batch_xs = train_data[start_idx:end_idx, :]
            batch_ys = train_label[start_idx:end_idx, :]

            row_rand_array = np.arange(test_data.shape[0])
            np.random.shuffle(row_rand_array)
            batch_xs_val = test_data[row_rand_array[0:batch_size], :]
            batch_ys_val = test_label[row_rand_array[0:batch_size], :]

            # loss
            _, loss_ = sess.run([train_op, loss], {x_placeholder: batch_xs, y: batch_ys, on_train: True, rate_place: 0.7})
            training_step += 1
            
            if training_step % display_step == 0:
                
                result, gt = sess.run([PD, GT], feed_dict={x_placeholder: batch_xs, y: batch_ys, on_train: False, rate_place: 1})
                train_accuracy = kappa(gt, result)

                result_val, gt_val = sess.run([PD, GT], feed_dict={x_placeholder: batch_xs_val, y: batch_ys_val, rate_place: 1,
                    on_train: False})
                
                valid_accuracy = kappa(gt_val, result_val)

                # test
                print('Epoch:', epoch, '| Step:', training_step, '| train loss: %.4f' % loss_,
                      '| train accuracy: %.4f' % train_accuracy,  '| valid accuracy: %.4f' % valid_accuracy,)
        # save
        saver.save(sess=sess, save_path='./model/prediction.ckpt')
