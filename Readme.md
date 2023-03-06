# Neural Style Transfer

----

Python/tensorflow implementation of Neural Style Transfer

----

Table of contents

 * [Installation](#installation)
 * [Running](#running)
 * [Pre-trained Data](#pre-trained-data)
 * [Architecture Documentation](#architecture-documentation)
 * [Use](#use)
 * [Additional Documentation](#additional-documentation)


## Installation

The following are required to run the application in secretshare.jar:
 1. python 3 (3.10.6)
 2. module "venv" if you use venvs
 3. python3 -m venv py3ml
 4. . ./py3ml/bin/activate
 5. pip install --upgrade pip
 6. pip install -r requirements.txt

TODO - install NVidia CUDA, update install instructions.

Note: not all the packages in requirements.txt are used by this project.  Instead, requirements.txt is a list of "generic machine learning packages". 

## Running

1. Activate the python virtual environment, then
```
    (py3ml) $ ./nst-standalone.py
  [creates directory outputs/output0051/*]
```

The default configuration takes about 2 minutes to complete (on relatively recent CPU, with no GPU/CUDA extensions installed).

## Pre-trained data

Also available from kaggle, from https://www.kaggle.com/datasets/kmader/full-keras-pretrained-no-top.  It a file that is part of a 1-gigabyte download, though.

This is used by tf.keras.applications.VGG19, see https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19
and https://keras.io/api/applications/#usage-examples-for-image-classification-models



## Architecture Documentation

See tensorflow tutorial at https://www.tensorflow.org/tutorials/generative/style_transfer


## Use

   See nsg-standalone.py, at the top, variable 'inputJson' to find the configuration
   parameters that are used each run.  In order of importance aka which ones are you likely to
   change:
   1. content_iamge_filename
   2. style_image_filename
   3. epochs
   4. save_epoch_every and print_epoch_every
   Actual MM neural net parameters:
   1. adam_learning_rate
   2. alpha (content) and beta (style)
   3. style_layers
   





## Additional Documentation

[Original Sourceforge Secret Sharing in Java] - original SCM location.  Out-of-date.

[Resources] - more links to useful Shamir Secret Share documentation and projects


[Original Sourceforge Secret Sharing in Java]:http://secretsharejava.sourceforge.net/
[Resources]:extrastuff/resources.md
[SecretShare1.4.1]:http://mvnrepository.com/artifact/com.tiemens/secretshare/1.4.1
[SecretShare1.4.2]:http://mvnrepository.com/artifact/com.tiemens/secretshare/1.4.2
[SecretShare1.4.3]:http://mvnrepository.com/artifact/com.tiemens/secretshare/1.4.3
[SecretShare 1.4.4]:http://mvnrepository.com/artifact/com.tiemens/secretshare/1.4.4
[SecretShare 1.4.4 Release Tag]:https://github.com/timtiemens/secretshare/releases/tag/v1.4.4
[SecretShare 1.4.4 Maven Central]:http://mvnrepository.com/artifact/com.tiemens/secretshare/1.4.4
