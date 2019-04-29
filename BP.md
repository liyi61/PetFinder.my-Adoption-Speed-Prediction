---
layout: page
title: Back Propogation Neural Network

---

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

To aovid overfitting, I did randomly dropout some nodes in the net
```python
def Dropout(x, rate):
    with tf.name_scope('Dropout_layer'):
        return tf.nn.dropout(x, rate)
```
Used Cross Entropy as our loss.
```python
with tf.name_scope('Loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
```

![bp2](/img/Screen Shot 2019-04-28 at 7.34.53 PM.png)

![bp1](/img/Screen Shot 2019-04-28 at 8.14.14 PM.png)

The training kappa for Back Propagation is 0.6940


## Testing Result

![testkappa](/img/testbp.png)

Testing Kappa is __0.6466__.  



