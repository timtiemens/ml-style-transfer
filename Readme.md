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

Note: not all the packages in requirements.txt are used by this project.  Instead, the requirements.txt is a superset of "machine learning packages" used by various machine-learning projects. 

## Running

1. Activate the python virtual environment, then
```
    (py3ml) $ ./nst-standalone.py
  [creates directory outputs/output0051/*]
```

The default configuration takes about 2 minutes to complete (on relatively recent CPU, with no GPU/CUDA extensions installed).

## Pre-trained data

Also available from [kaggledataset].  It a file that is part of a 1-gigabyte download, though.

This is used by tf.keras.applications.VGG19, see [tensorflowVGG19].
and [kerasimageclassificaton].



## Architecture Documentation

See tensorflow tutorial at https://www.tensorflow.org/tutorials/generative/style_transfer


## Use

   See nsg-standalone.py, at the top, variable 'inputJson' to find the configuration
   parameters that are used each run.  In order of importance aka which ones are you likely to
   change:
   1. content_image_filename
   2. style_image_filename
   3. epochs
   4. save_epoch_every and print_epoch_every

   Actual neural net parameters:
   1. adam_learning_rate
   2. alpha (content) and beta (style)
   3. style_layers


## Timings

1. Intel i5-12600 6 cores, 12 vcores, 3.3 GHz
    1. Windows Subsystem Linux, Ubuntu 22.04
        1. Standard (250 epoch, louvre, monet) - 138 seconds (180 seconds elapsed)
    2. Windows 11, cmd shell
        1. Standard (250 epoch, louvre, monet) - 201 seconds (210 seconds elapsed)
2. Intel Xeon E5-2640 12 cores, 2.5Ghz, 64 GB RAM, CentOS, VMworkstation
    1. Virtual Machine, Ubuntu 22.04, 4 cores, 8 GB RAM
        1. Standard (250 epoch, louvre, monet) - 1010 seconds (1050 seconds elapsed)
3. 2x Intel Xeon, E5-2680, 20 cores, 40 vcores, 2.8Ghz, 256 GB RAM, Windows, VMworkstation
    1. Virtual Machine, Ubuntu 22.04, 8 cores, 16 GB RAM
        1. Standard (250 epoch, louvre, money) -  599 seconds (605 seconds elapsed)



## Additional Documentation

[Resources] - TBD


[kaggledataset]:https://www.kaggle.com/datasets/kmader/full-keras-pretrained-no-top
[tensorflowVGG19]:https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19
[kerasimageclassification]:https://keras.io/api/applications/usage-examples-for-image-classification-models

[Resources]:extrastuff/resources.md
