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
 * [Standard Run](#standard-run)
 * [More Examples](#more-examples)
 * [Timings](#timings)
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

The pre-trained [vgg19 weights .h5 file](pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5) is in this repository as an 77 MB file.

It is also available from [kaggledataset].  To get it, you have to download the entire 1 GB download, though.

This is used by tf.keras.applications.VGG19, see [tensorflowVGG19].
and [kerasvgg19].



## Architecture Documentation

See tensorflow tutorial at https://www.tensorflow.org/tutorials/generative/style_transfer


## Use

   See [nst-standalone.py](nst-standalone.py), at the top, variable 'inputJson' to find the configuration parameters that are used each run.
   In order of importance aka the ones that you are most likely to change:
   1. content_image_filename --content
   2. style_image_filename --style
   3. epochs --epochs
   4. save_epoch_every and print_epoch_every --saveEveryEpoch

   Actual neural net parameters:
   1. adam_learning_rate --learningRate
   2. alpha (content) and beta (style) weights  --alpha and --beta
   3. style_layers
   4. random generator seed --seed

## Standard Run

The standard run uses the louvre image for content and the monet image for
style, and runs 250 epochs to create the output image:

| Type |  |
| --- | ----------- |
| Content | <img src="images/louvre_small.jpg" width=400 height=400> |
| Style   | <img src="images/monet.jpg" width=400 height=400> |
| Output  | <img src="images/sample_louvre_monet_250.jpg" width=400 height=400> |
| (this is Output at 2500 epochs)  | <img src="images/sample_louvre_monet_2500.jpg" width=400 height=400> |

## More Examples

See [ml-style-transfer-samples](https://github.com/timtiemens/ml-style-transfer-samples) for a more comprehensive list of example input.json files,
and different style images and different content (base) images.





## Timings

**seconds** - from Python "finished" output

**seconds elapsed** - wall-clock or "time" output

1. Intel i5-12600 6 cores, 12 vcores, 3.3 GHz
    1. Windows Subsystem Linux, Ubuntu 22.04
        1. Standard (250 epoch, louvre, monet) - 138 seconds (180 seconds elapsed)
    2. Windows 11, cmd shell
        1. Standard (250 epoch, louvre, monet) - 201 seconds (210 seconds elapsed)
    3. Windows 11, miniconda, CUDA 11.2, cudnn 8.1.0, RTX 3080Ti GPU
        1. Standard (250 epoch, louvre, monet) -  12 seconds (16 seconds elapsed)
    4. Windows Subsystem Linux, Ubuntu 22.04, tensorflow 2.12.0, cudatoolkit 11.8.0, RTX 3080Ti GPU
        1. Standard (250 epoch, louvre, monet) -  9 seconds (11 seconds elapsed)

2. Intel Xeon E5-2640 12 cores, 2.5Ghz, 64 GB RAM, CentOS, VMworkstation
    1. Virtual Machine, Ubuntu 22.04, 4 cores, 8 GB RAM
        1. Standard (250 epoch, louvre, monet) - 1010 seconds (1050 seconds elapsed)
3. 2x Intel Xeon, E5-2680, 20 cores, 40 vcores, 2.8Ghz, 256 GB RAM, Windows, VMworkstation
    1. Virtual Machine, Ubuntu 22.04, 8 cores, 16 GB RAM
        1. Standard (250 epoch, louvre, money) -  599 seconds (605 seconds elapsed)
4. AWS p3.2xlarge, [AWS p3](https://aws.amazon.com/ec2/instance-types/p3/), Intel Xeon Skylake 8175, 2.5 GHz, 8 vCPU, 61 GB RAM, 1 Tesla V100 GPU, $3.06/hour
    1. ami-0649417d1ede3c91a, Ubuntu 20.04, tensorflow 2.12.0
        1. Standard (250 epoch, louvre, monet) - 11 seconds (14 seconds elapsed)

5. AWS t2.large, [AWS t2](https://aws.amazon.com/ec2/instance-types/t2/), Intel Xeon E5-2686 v4, 2.30GHz, 2 vCPU, 8 GB RAM, no GPU, $0.093/hour
    1. ami-0649417d1ede3c91a, Ubuntu 20.04
        1. Standard (250 epoch, louvre, monet) - 845 seconds (849 seconds elapsed)
6. AWS c5.4xlarge, [AWS c5](https://aws.amazon.com/ec2/instance-types/c5/), Intel Platinum 8223CL, 3.0 GHz, 16 vCPU, 32 GB RAM, no GPU, $0.68/hour
    1. ami-0649417d1ede3c91a, Ubuntu 20.04
        1. Standard (250 epoch, louvre, monet) - 219 seconds (221 seconds elapsed)

7. AWS inf1.xlarge, [AWS inf1](https://aws.amazon.com/ec2/instance-types/inf1/), Intel Xeon 8275CL, 3.0 GHz, 4 vCPU, 8 GB RAM, no GPU, $0.228/hour
    1. ami-0649417d1ede3c91a, Ubuntu 20.04, See FootNote1
        1. Standard (250 epoch, louvre, monet) - 484 seconds (490 seconds elapsed)

FootNote1 - ami-0649417d1ede3c91a is probably the wrong AMI to use with this instance type.


## Additional Documentation

[Resources] - TBD


[kaggledataset]:https://www.kaggle.com/datasets/kmader/full-keras-pretrained-no-top
[tensorflowVGG19]:https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19
[kerasvgg19]:https://keras.io/api/applications/vgg/#vgg19-function

[Resources]:extrastuff/resources.md
