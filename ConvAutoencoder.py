"""
TF Convolutional Autoencoder

Arash Saber Tehrani - May 2017
Reference: https://github.com/arashsaber/Deep-Convolutional-AutoEncoder

Modified David Yu - July 2018
Reference: https://github.com/MrDavidYu/TF_Convolutional_Autoencoder
Add ons:
1. Allows for custom .jpg input
2. Checkpoint save/restore
3. TensorBoard logs for input/output images
3. Input autorescaling
4. ReLU activation replaced by LeakyReLU

"""
import os
import re
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

# Some important consts
num_examples = 669
batch_size = 30
n_epochs = 500
save_steps = 500  # Number of training batches between checkpoint saves

checkpoint_dir = "./ckpt/"
model_name = "ConvAutoEnc.model"
logs_dir = "./logs/run1/"

# Fetch input data (faces/trees/imgs)
data_dir = "./data/celebF/"
data_path = os.path.join(data_dir, '*.jpg')
data = glob(data_path)

if len(data) == 0:
    raise Exception("[!] No data found in '" + data_path+ "'")

def path_to_img(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

np.random.shuffle(data)
imread_img = path_to_img(data[0])  # test read an image

if len(imread_img.shape) >= 3: # check if image is a non-grayscale image by checking channel number
    c_dim = path_to_img(data[0]).shape[-1]
else:
    c_dim = 1

is_grayscale = (c_dim == 1)

# tf Graph Input
# face data image of shape 84*84=7056 N.B. originally without the depth 3
x = tf.placeholder(tf.float32, [None, 42, 42, 3], name='InputData')

if __debug__:
    print("Reading input from:" + data_dir)
    print("Input image shape:" + str(imread_img.shape))
    print("Assigning input tensor of shape:" + str(x.shape))
    print("Writing checkpoints to:" + checkpoint_dir)
    print("Writing TensorBoard logs to:" + logs_dir)

"""
We start by creating the layers with name scopes so that the graph in
the tensorboard looks meaningful
"""


# strides = [Batch, Height, Width, Channels]  in default NHWC data_format. Batch and Channels
# must always be set to 1. If channels is set to 3, then we would increment the index for the
# color channel by 3 everytime we convolve the filter. So this means we would only use one of
# the channels and skip the other two. If we change the Batch number then it means some images
# in the batch are skipped.
#
# To calculate the size of the output of CONV layer:
# OutWidth = (InWidth - FilterWidth + 2*Padding)/Stride + 1
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.variable_scope(name):
        W = tf.get_variable(name='w_' + name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.leaky_relu(out)
        return out


# tf.contrib.layers.conv2d_transpose, do not get confused with 
# tf.layers.conv2d_transpose
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.variable_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                 num_outputs= n_outputs,
                 kernel_size=kshape,
                 stride=strides,
                 padding='SAME',
                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                 activation_fn=tf.nn.leaky_relu)
        return out

# Input to maxpool: [BatchSize, Width1, Height1, Channels]
# Output of maxpool: [BatchSize, Width2, Height2, Channels]
#
# To calculate the size of the output of maxpool layer:
# OutWidth = (InWidth - FilterWidth)/Stride + 1
#
# The kernel kshape will typically be [1,2,2,1] for a general 
# RGB image input of [batch_size,64,64,3]
# kshape is 1 for batch and channels because we don't want to take
# the maximum over multiple examples of channels.

def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(x,
                 ksize=kshape, #size of window
                 strides=strides,
                 padding='SAME')
        return out


def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def fullyConnected(input, name, output_size):
    with tf.variable_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size)) # get total num of cells in one input image
        W = tf.get_variable(name='w_'+name,
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                shape=[output_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
        return out


def dropout(input, name, keep_rate):
    with tf.variable_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out


def ConvAutoEncoder(x, name, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        """
        We want to get dimensionality reduction of 11664 to 44656
        Layers:
            input --> 84, 84 (7056)
            conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
            pool1 --> 42, 42, 25
            dropout1 --> keeprate 0.8
            reshape --> 42*42*25
            FC1 --> 42*42*25, 42*42*5
            dropout2 --> keeprate 0.8
            FC2 --> 42*42*5, 8820 --> output is the encoder vars
            FC3 --> 8820, 42*42*5
            dropout3 --> keeprate 0.8
            FC4 --> 42*42*5,42*42*25
            dropout4 --> keeprate 0.8
            reshape --> 42, 42, 25
            deconv1 --> kernel size:(5,5,25), n_filters: 25
            upsample1 --> 84, 84, 25
            FullyConnected (outputlayer) -->  84* 84* 25, 84 * 84 *  1
            reshape --> 84 * 84
        """
        input = tf.reshape(x, shape=[-1, 42, 42, 3])

        # coding part
        c1 = conv2d(input, name='c1', kshape=[7, 7, 3, 25])  # kshape = [k_h, k_w, in_channels, out_chnnels]
        p1 = maxpool2d(c1, name='p1')
        do1 = dropout(p1, name='do1', keep_rate=0.75)
        do1 = tf.reshape(do1, shape=[-1, 21*21*25])  # reshape to 1 dimensional (-1 is batch size)
        fc1 = fullyConnected(do1, name='fc1', output_size=21*21*5)
        do2 = dropout(fc1, name='do2', keep_rate=0.75)
        fc2 = fullyConnected(do2, name='fc2', output_size=21*21*3)
        # Decoding part
        fc3 = fullyConnected(fc2, name='fc3', output_size=21 * 21 * 5)
        do3 = dropout(fc3, name='do3', keep_rate=0.75)
        fc4 = fullyConnected(do3, name='fc4', output_size=21 * 21 * 25)
        do4 = dropout(fc4, name='do3', keep_rate=0.75)
        do4 = tf.reshape(do4, shape=[-1, 21, 21, 25])
        dc1 = deconv2d(do4, name='dc1', kshape=[7,7],n_outputs=25)
        up1 = upsample(dc1, name='up1', factor=[2, 2])
        output = fullyConnected(up1, name='output', output_size=42*42*3)
        # print("output.shape"+str(output.shape))
        # print("x.shape"+str(x.shape))
        with tf.variable_scope('cost'):
            # N.B. reduce_mean is a batch operation! finds the mean across the batch
            cost = tf.reduce_mean(tf.square(tf.subtract(output, tf.reshape(x,shape=[-1,42*42*3]))))
        return x, tf.reshape(output,shape=[-1,42,42,3]), cost # returning, input, output and cost
#   ---------------------------------

def train_network(x):

    with tf.Session() as sess:

        _, _, cost = ConvAutoEncoder(x, 'ConvAutoEnc')
        with tf.variable_scope('opt'):
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("cost", cost)
        tf.summary.image("face_input", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True)[0], max_outputs=4)
        tf.summary.image("face_output", ConvAutoEncoder(x, 'ConvAutoEnc', reuse=True)[1], max_outputs=4)
        merged_summary_op = tf.summary.merge_all()  # Merge all summaries into a single op

        sess.run(tf.global_variables_initializer())  # memory allocation exceeded 10% issue

        # Model saver
        saver = tf.train.Saver()

        counter = 0  # Used for checkpointing
        success, restored_counter = restore(saver, sess)
        if success:
            counter = restored_counter
            print(">>> Restore successful")
        else:
            print(">>> No restore checkpoints detected")        

        # create log writer object
        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

        for epoch in range(n_epochs):
            avg_cost = 0
            n_batches = int(num_examples / batch_size)
            # Loop over all batches
            for i in range(n_batches):
                counter += 1
                print("epoch " + str(epoch) + " batch " + str(i))

                batch_files = data[i*batch_size:(i+1)*batch_size]  # get the current batch of files
                # TODO: add functionality to autorescale
                # batch = [
                # get_image(batch_file,
                #         input_height=42,
                #         input_width=42,
                #         resize_height=42,
                #         resize_width=42,
                #         crop=True,
                #         grayscale=False) for batch_file in batch_files] # get_image will get image from file dir after applying resize operation. 
                batch = [path_to_img(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Get cost function from running optimizer
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_images})

                # Compute average loss
                avg_cost += c / n_batches

                writer.add_summary(summary, epoch * n_batches + i)

                if counter % save_steps == 0:
                    save(saver, counter, sess)

            # Display logs per epoch step
            print('Epoch', epoch + 1, ' / ', n_epochs, 'cost:', avg_cost)

        print('>>> Optimization Finished')


# Create checkpoint
def save(saver, step, session):
    print(">>> Saving to checkpoint, step:" + str(step))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(session,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

# Restore from checkpoint
def restore(saver, session):
    print(">>> Restoring from checkpoints...")
    checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
      checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
      saver.restore(session, os.path.join(checkpoint_dir, checkpoint_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",checkpoint_name)).group(0))
      print(">>> Found restore checkpoint {}".format(checkpoint_name))
      return True, counter
    else:
      return False, 0

train_network(x)