#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint
import time
import datetime
import json
import re
#%matplotlib inline

# version of this script:
version = 'v1.2'

# AVAILABLE INPUTS:
availableJson = {
    'content' : {
        'louvre'     : 'images/louvre_small.jpg'
    },
    'style' : {
        'monet'      : 'images/monet.jpg',
        'katsushika' : 'images/katsushika.jpg',
        'kuniyoshi' : 'images/kuniyoshi.jpg',
        'sandstone' : 'images/sandstone.jpg',
        'stones'    : 'images/stone_style.jpg'
    }
}


# INPUTS:
inputJson = {
    'image_size' : 400,
    'vgg_weights_filename' : 'pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',


    #'epochs': 2501,    # takes 1600 seconds
    'epochs' : 251,     # takes  137 seconds
    'print_epoch_every':  5,
    'save_epoch_every': 250,
    'output_dir_top': 'outputs',
    'output_dir_this_run' : 'auto',   # ends up being e.g. 'output0159',


    'content_image_filename' : availableJson['content']['louvre'],
    'style_image_filename'   : availableJson['style']['monet'],

    'print_layers' : False,

    #'adam_learning_rate' : 0.01,  # original rate
    'adam_learning_rate' : 0.05,
    #'adam_learning_rate' : 0.1,   # very "impressionistic" at 250epochs
    # 'adam_learning_rate' : 0.5,   # too agressive after 250epochs

    # weights/ratio content versus style
    'alpha' : 10,   # content
    'beta'  : 40,   # style

    'tf.random.seed': 272,


    # STYLE_LAYERS = [ ]
    'style_layers' : [ 
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)],


#        ('block1_conv1', 0.1),
#        ('block2_conv1', 0.05),
#        ('block3_conv1', 0.5),
#        ('block4_conv1', 0.05),
#        ('block5_conv1', 0.3)],

    'content_image_filename_large': 'images/louvre.jpg',
}


#
# input_3
# block1_conv1
# block1_conv2
# block1_pool
# block2_conv1
# block2_conv2
# block2_pool
# block3_conv1
# block3_conv2
# block3_conv3
# block3_conv4
# block3_pool
# block4_conv1
# block4_conv2
# block4_conv3
# block4_conv4
# block4_pool
# block5_conv1
# block5_conv2
# block5_conv3
# block5_conv4
# block5_pool


def get_vgg(img_size, weights_input_file):

    vgg = tf.keras.applications.VGG19(include_top=False,
                                      input_shape=(img_size, img_size, 3),
                                      weights=weights_input_file)

    vgg.trainable = False

    return vgg


def get_image(img_filename):
    ret = Image.open(img_filename)
    return ret

def load_resize_image(img_filename, img_size):
    ret = np.array(get_image(img_filename).resize((img_size, img_size)))
    ret = tf.constant(np.reshape(ret, ((1,) + ret.shape)))

    return ret

def die(s):
    print(s)
    sys.exit(1)

def get_numbers_from_filenames(allfilenames, prefix, postfix):
    ret = []
    pattern = "^" + prefix + "(\d+)" + postfix + "$"
    thepattern = re.compile(pattern)
    for file in allfilenames:
        match = thepattern.match(file)
        if match:
            i = int(match.group(1))
            ret.append(i)

    return ret

def get_auto_output_dir_this_run(topdir, zeropaddigits=4,
                                 prefix='output', postfix=''):
    alllist = os.listdir(topdir)
    dirs = [f for f in alllist if os.path.isdir(topdir + "/" + f)]
    numbers = get_numbers_from_filenames(dirs, "output", "")
    numbers = sorted(numbers, key=int)
    if numbers and len(numbers) > 0:
        highest = numbers[-1]
    else:
        highest = 49
    #print("Highest is " + str(highest))
    nextnumber = highest + 2
    format_string = f'0{zeropaddigits}d'
    zeropad = f"{nextnumber:{format_string}}"
    retdirname = "" + prefix + zeropad + postfix
    return retdirname

def process_input_json(inputJson):
    """
    Perform all validation and processing on inputJson
    """
    # First: set inputJson['output_dir_this_run']
    #   Then - make sure it is not currently a directory, then make it
    # Second: check files exist: cotent_image_filename, etc.
    # Third set the random seed if found in inputJson

    dirtop = inputJson['output_dir_top']
    if not os.path.isdir(dirtop):
        os.makedirs(dirtop)
    current = inputJson['output_dir_this_run']
    if current.startswith('auto'):
        proposed_out_dir = get_auto_output_dir_this_run(dirtop, 4)
        fullpath = dirtop + "/" + proposed_out_dir
        if os.path.isdir(fullpath):
            die(f"Refusing to create output directory {fullpath}, programmer error")
        else:
            inputJson['output_dir_this_run'] = proposed_out_dir
            current = inputJson['output_dir_this_run']
            print(f"NOTE: output_dir_this_run is {current}")
    fullpath = dirtop + "/" + current
    if os.path.isdir(fullpath):
        die(f"Refusing to create output directory {fullpath}")
    else:
        os.makedirs(fullpath)
        if not os.path.isdir(fullpath):
            die(f"Internal error creating directory {fullpath}")

    keys_for_filenames = [ 'content_image_filename', 'style_image_filename' ]
    for key in keys_for_filenames:
        filename = inputJson[key]
        if not os.path.isfile(filename):
            die(f"File not found: '{filename}' for key '{key}'")

    if 'tf.random.seed' in inputJson:
        setit = inputJson['tf.random.seed']
        print(f"NOTE: setting seed to {setit}")
        tf.random.set_seed(setit)



def content_compute_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]


    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

    # compute the cost with tensorflow
    J_content = ( 1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum( tf.square( tf.subtract(a_C, a_G)))

    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    #(≈1 line)
    GA = tf.matmul( A, tf.transpose(A))

    return GA

def layer_style_compute_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W)
    a_S = tf.transpose( tf.reshape(a_S, shape=[n_H * n_W, n_C]) )
    a_G = tf.transpose( tf.reshape(a_G, shape=[n_H * n_W, n_C]) )

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    #    Just FYI:    don't use tf.square()
    J_style_layer =  1 / ( 4 * n_C * n_C * n_H * n_W * n_H * n_W)
    J_style_layer =  J_style_layer * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))


    return J_style_layer


def style_compute_cost(style_image_output, generated_image_output, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                   - the names of the layers we would like to extract style from
                   - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = layer_style_compute_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


def total_content_style_compute_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    # Total cost for content and style, scaled by alpha and beta
    J = alpha * J_content + beta * J_style


    return J




def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def clip_0_to_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def create_generated_image(base_image):
    generated_image = tf.Variable(tf.image.convert_image_dtype(base_image,
                                                               tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = clip_0_to_1(generated_image)


    return generated_image


@tf.function()
def train_tape_step(generated_image, optimizer, alpha, beta, style_layers):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images
        #      a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image

        a_G = vgg_model_outputs(generated_image)

        # Compute the style cost

        J_style = style_compute_cost(a_S, a_G, style_layers)


        # Compute the content
        J_content = content_compute_cost(a_C, a_G)
        
        # Compute the total cost
        J = total_content_style_compute_cost(J_content, J_style, alpha = alpha, beta = beta)


    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_to_1(generated_image))

    # unused
    return J




if __name__ == '__main__':
    #
    # validate and pre-process everything in inputJson configuration:
    #
    process_input_json(inputJson)
    print("Input Json process complete")

    time_start = int(time.time())    
    vgg = get_vgg(inputJson['image_size'],
                  inputJson['vgg_weights_filename'])
    # TODO: why is this such a bland output?
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vgg)


    if False:
        content_image = get_image(inputJson['content_image_filename_large'])
        content_image    # display?  not in WSL   ?? in Ubun22

    if inputJson['print_layers']:
        for layer in vgg.layers:
            print(layer.name)

    content_image = load_resize_image(inputJson['content_image_filename'],
                                      inputJson['image_size'])
    print(content_image.shape)
    #imshow(content_image[0])
    #plt.show()

    style_image = load_resize_image(inputJson['style_image_filename'],
                                    inputJson['image_size'])
    print(style_image.shape)

    generated_image = create_generated_image(content_image)
    print(generated_image.shape)

    STYLE_LAYERS = inputJson['style_layers']

    content_layer = [('block5_conv4', 1)]   # the last conv layer

    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets  = vgg_model_outputs(style_image)    # Style encoder

    
    # Assign the content image to be the input of the VGG model.  
    # Set a_C to be the hidden layer activation from the layer we have
    # selected
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image,
                                                                     tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    
    # Assign the input of the model to be the "style" image 
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image,
                                                                   tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    optimizer = tf.keras.optimizers.Adam(learning_rate=inputJson['adam_learning_rate']) 

    generated_image = tf.Variable(generated_image)

    output_dir = inputJson['output_dir_top'] + "/" + inputJson['output_dir_this_run']
    with open(output_dir + "/input.json", "w") as f:
        f.write(json.dumps(inputJson, indent=4))
    
    numberofdigits = 5   # e.g. image_12345.jpg or image_00012.jpg
    format_string = f'0{numberofdigits}d'

    # Save/Show the generated image at some epochs
    epochs = inputJson['epochs']
    for i in range(epochs):
        train_tape_step(generated_image, optimizer,
                   inputJson['alpha'], inputJson['beta'],
                   STYLE_LAYERS)
        if i % inputJson['print_epoch_every'] == 0:
            print(f"Epoch {i} ")
        if i % inputJson['save_epoch_every'] == 0:
            image = tensor_to_image(generated_image)
            image.save(f"{output_dir}/image_{i:{format_string}}.jpg")
            imshow(image)            
            #plt.show() 


    time_end = int(time.time())
    elapsed = time_end - time_start
    print(f"Epoch -- finished (elapsed seconds={elapsed}) output={output_dir}")
    outputJson = {
        'elapsed_seconds': elapsed,
        'time_start': time_start,
        'time_end': time_end,
        'timestamp': datetime.datetime.fromtimestamp(time_start).isoformat(),
        'nst_standalone_version': version,
    }
    with open(output_dir + "/output.json", "w") as f:
        f.write(json.dumps(outputJson, indent=4))    
