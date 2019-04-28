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

![bp2](/img/Screen Shot 2019-04-28 at 3.32.37 AM.png)
![bp1](/img/Screen Shot 2019-04-28 at 1.10.40 PM.png)



## Testing Result


