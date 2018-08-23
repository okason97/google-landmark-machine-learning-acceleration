
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os


# In[2]:


# Hyperparameters

batch_size = 32
learning_rate = 0.01
training_epochs = 10
display_step = 2

# To prevent overfitting
dropout = 0.75


# In[3]:


# Dataset and iterator creation

in_dir = "../input/normalized-images/"
sample_size = 20

data = pd.read_csv('../input/train.csv')
available_list = np.array([os.path.splitext(filename)[0] for filename in os.listdir(in_dir)])
data = data[data["id"].isin(available_list)]
data = data.groupby('landmark_id', group_keys=False).apply(lambda df: df.sample(sample_size, random_state=123))
full_url = np.vectorize(lambda x: in_dir+x+".jpg")
filenames = full_url(data["id"].values)
labels = pd.get_dummies(data["landmark_id"]).values
train_filenames, test_filenames = filenames[:int(filenames.shape[0]*0.75)], filenames[int(filenames.shape[0]*0.75):]
train_labels, test_labels = labels[:int(labels.shape[0]*0.75)], labels[int(labels.shape[0]*0.75):]

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_decoded.set_shape((256, 256, 1))
    return image_decoded, label

train_data = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_data = train_data.shuffle(buffer_size=10000)

# for a small batch size
train_data = train_data.map(_parse_function, num_parallel_calls=4)
train_data = train_data.batch(batch_size)

# for a large batch size (hundreds or thousands)
# dataset = dataset.apply(tf.contrib.data.map_and_batch(
#     map_func=_parse_function, batch_size=batch_size))

# with gpu usage
# train_data = train_data.prefetch(1)

test_data = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_data = test_data.map(_parse_function, num_parallel_calls=4)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
# iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

train_init = iterator.make_initializer(train_data) # Inicializador para train_data
test_init = iterator.make_initializer(test_data) # Inicializador para test_data

# Total ammount of landmarks
n_landmarks = len(data.groupby("landmark_id")["landmark_id"])


# In[4]:


# Placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_landmarks])

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add        (tf.nn.conv2d(img, w,        strides=[1, 1, 1, 1],        padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img,         ksize=[1, k, k, 1],        strides=[1, k, k, 1],        padding='SAME')

# weights and bias conv layer 1
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

# conv layer
conv1 = conv2d(x,wc1,bc1)

# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 128*128 matrix.
conv1 = max_pool(conv1, k=2)

# dropout to reduce overfitting
keep_prob = tf. placeholder(tf.float32)
conv1 = tf.nn.dropout(conv1,keep_prob)

# weights and bias conv layer 2
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

# conv layer
conv2 = conv2d(conv1,wc2,bc2)

# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
conv2 = max_pool(conv2, k=2)

# dropout to reduce overfitting
conv2 = tf.nn.dropout(conv2, keep_prob)

# weights and bias fc 1
wd1 = tf.Variable(tf.random_normal([64*64*64, 256]))
bd1 = tf.Variable(tf.random_normal([256]))

# fc 1
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1))
dense1 = tf.nn.dropout(dense1, keep_prob)

# weights and bias out
wout = tf.Variable(tf.random_normal([256, n_landmarks]))
bout = tf.Variable(tf.random_normal([n_landmarks]))

# prediction
pred = tf.add(tf.matmul(dense1, wout), bout)

# softmax
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[5]:


# Session start

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Required to get the filename matching to run.
    sess.run(init)
    
    step = 1
    # Compute epochs.
    for i in range(training_epochs):
        print("epoch: {}".format(i))
        sess.run(train_init)
        try:
            while True:
                batch_xs, batch_ys = sess.run(next_element)
                                
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout}) 
                                
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,  keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    print("step: {}".format(step))
                    print("accuracy: {}".format(acc))
                    print("loss: {}".format(loss))
                    print("\n")
                step += 1
        except tf.errors.OutOfRangeError:
            pass
        
    step = 1
    # Test
    sess.run(test_init)
    try:
        while True:
            batch_xs, batch_ys = sess.run(next_element)
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,  keep_prob: 1.})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("test \n")
                print("accuracy: {}".format(acc))
                print("loss: {}".format(loss))
                print("\n")
            step += 1
    except tf.errors.OutOfRangeError:
        pass

