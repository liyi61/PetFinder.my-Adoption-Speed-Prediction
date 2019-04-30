---
layout: page
title: Back Propogation Neural Network

---
The neural network contained three ResNet(convolutional) blocks(each including 2 convolutional layers) and four fully-connected layers including the output layer. The ResNet blocks were used to learn the features. Then the features were flattened into vector forms in order to feed to the fully connected layer to do the prediction. 

## Training Parameters Setting

```python
num_examples_per_epoch_for_train = train_data.shape[0]
training_epochs = 70
batch_size = 1200
display_step = 1
LR = 0.0001
```
![parameter](/img/setting.png)

## Training Process and Reuslt Kappa

To aovid overfitting, some nodes were randomly droped out in the network.

```python
def Dropout(x, rate):
    with tf.name_scope('Dropout_layer'):
        return tf.nn.dropout(x, rate)
```
Cross Entropy was used as loss function.
```python
with tf.name_scope('Loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
```

![bp2](/img/Screen Shot 2019-04-28 at 7.34.53 PM.png)

![bp1](/img/Screen Shot 2019-04-28 at 8.14.14 PM.png)

The training kappa for Back Propagation Neural Network is __0.6940__.
The Training took about 6 hours.

## Testing Result

![testkappa](/img/testbp.png)

Testing Kappa is __0.6466__.  
There is a hugh improvement!



