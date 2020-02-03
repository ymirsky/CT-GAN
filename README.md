# Overview
In this repository you will find a Keras implementation of CT-GAN: A framework for adding or removing evidence in 3D volumetric medical scans. In this readme, you will find a description of CT-GAN, examples of how to use the code, and links to our tampered datasets. For more details, please see our publication:

*Yisroel Mirsky, Tom Mahler, Ilan Shelef, and Yuval Elovici. CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning. 28th USENIX Security Symposium (USENIX Security 19)*

([full paper here](https://www.usenix.org/system/files/sec19-mirsky_0.pdf))
Links to datasets are found below.
For access to the pretrained models, please reach out to me (contact below). We will only supply the models to verified academic researchers.

**Disclaimer**: This code has been published for research purposes only. It is our hope that with this code, others will be able to better understand this threat and find better ways to mitigate it.
 

## What is CT-GAN?

In 2018, clinics and hospitals were hit with numerous attacks
leading to significant data breaches and interruptions in
medical services. An attacker with access to medical imagery can alter the
contents to cause a misdiagnosis. Concretely, the attacker can
add or remove evidence of some medical condition. The figure below illustrates this attack vector.

*An illustration of the attack vector within a hostpital:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/attackvec.png)

There are many reasons why an attacker would want to
alter medical imagery: to disrupt a [political] leader's life, perform ransomware, an act of insurance fraud, falsifying research evidence, sabotaging another company’s research, job theft,
terrorism, assassination, and even murder.

CT-GAN is a framework for automatically injecting and removing medical evidence from 3D medical scans such as those produced from CT and MRI. The framework consists of two conditional GANs
(cGAN) which perform in-painting (image completion) on
3D imagery. For injection, a cGAN is trained on unhealthy
samples so that the generator will always complete the images
accordingly. Conversely, for removal, another cGAN is trained
on healthy samples only.
To make the process efficient and the output anatomically
realistic, CT-GAN perform the following steps when tampering a scan, the framework
1. locates where the evidence should be inject/removed, 
2. cuts out a rectangular cuboid from the location, 
3. interpolates (scales) the cuboid to 1:1:1 ratio,
4. modifies the cuboid with the respective cGAN,
5. rescales the cuboid back to the original ratio,
6. pastes the scaled and tampered cuboid back into the original scan

*Top: the complete cancer injection/removal process. Bottom: sample images from the injection process. The grey numbers indicate from which step the image was taken. The sample 2D images are the middle slice of the respective 3D cuboid:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/pipeline.png)

By dealing with a small portion of the scan, the problem complexity is reduced by focusing
the GAN on the relevant area of the body (as opposed to the entire CT). Moreover, the algorithm complexity is reduced
by processing fewer inputs (voxels) and concepts (anatomical features). This results in fast execution and high anatomical realism. In our paper we show how CT-GAN can trick expert radiologists 98% percent of the time and a state-of-the-art AI 100% of the time (in the case of lung cancer).

## The cGAN (pix2pix) architecture
The cGAN architecture (layers and configurations) used for training the injector and remover generator networks is illustrated below. Overall, each cGAN has 189.5 million trainable parameters each.

*The network architecture, layers, and parameters used for both the injection and removal GAN networks:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/arch.png)


## Sample results
*Top: 3D models of injection (left) and removal (right) of a cancerous pulmonary lung nodule. Bottom: sample injections (left) and removals (right), where for each image, the left side is before tampering and the right side is after and only the middle 2D slice is shown:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/cancersamples.png)

*CT-GAN used to inject brain tumors into MRIs of healthy brains. Top: context, middle: in-painted result, and bottom: ground-truth. Showing one slice
in a 64x64x16 cuboid:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/braintrain.png)

*Demo video:*
[![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/demovid.png)](https://youtu.be/_mkRAArj-x0)


## This version's features and limitations
**Features**

* build normalized/preprocessed training dataset from mhd/raw and dicom medical scans.
* train the injection and removal networks 
* inject and remove evidence from mhd/raw and dicom scans

**Limitations**

* this version will not automatically locate candicate injection/removal locations within a target scan. Please see our paper for details on this algorithm.


# The CT-GAN Code

The code has been written with OOP and enables you to train CT-GAN for injection and/or removal. This repo contains example scripts for perfoming every step of CT-GAN, and the primary source code (found in the 'procedures' directory). To configure CT-GAN and its inputs, you must change the contents of [config.py](config.py) accordingly (see below for details)
Example scripts for running CT-GAN are in the main directory:
* **1A_build_injector_trainset.py**	: Builds a preprocessed training dataset from a set of medical scans for the purpose of injecting evidence. 
* **1B_build_remover_trainset.py**	: Builds a preprocessed training dataset from a set of medical scans for the purpose of removing evidence. 
* **2A_train_injector.py**			: Trains the injection cGAN to perform in-painting using the extracted dataset. 
* **2B_train_remover.py**			: Trains the removal cGAN to perform in-painting using the extracted dataset. 
* **3A_inject_evidence.py**			: Injects evidence into a given scan at the given coordinates (you must change the hard-coded paths and values in script). 
* **3B_remove_evidence.py**			: Removes evidence from a given scan at the given coordinates (you must change the hard-coded paths and values in script). 
* **tamper.py**						: An all-in-one command-line tool for tampering scans given the trained model(s). 
* **GUI.py**						: An interactive GUI for point and click scan tampering 

## Implementation Notes: 

* Tested on Windows Server 2012 R2 with 256GB RAM: using two Intel Xeon CPUs (E5-2660 v4 with 28 Logical Processor(s))
* Tested on an Ubuntu v4.4.0-142 with 128GB RAM and Xeon E7 CPUs (16 cores): using one Nvidia Titan X Pascal (Driver 418.46, CUDA 10.1)
* Tested using Anaconda 3.7.3, Keras with the tensorflow back-end v1.13.1
* Python dependencies: 
	* Common in most installations: multiprocessing, joblib, itertools, numpy, pickle
	* What you may need to install: keras, tensorflow, SimpleITK, pydicom, scipy, pandas, matplotlib

To install the dependencies, run this in the terminal:
```
pip install --upgrade scipy matplotlib pandas tensorflow keras SimpleITK pydicom
```
 
## Coordinate Systems
Coordinates in a medical scans can be denoted using world coordinates or image (voxel) coordinates. In order to use CT-GAN you need to be familiar with the difference between these systems. [[source](https://www.slicer.org/wiki/Coordinate_systems)]

![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/coordinate_systems.png)

### World Coordinate System
The world coordinate system is typically a Cartesian coordinate system in which a model (e.g. a MRI scanner or a patient) is positioned. Every model has its own coordinate system but there is only one world coordinate system to define the position and orientation of each model.
### Image (Voxel) Coordinate System
A voxel represents a value on a regular grid in three-dimensional space (like 3D pixels). The voxel coordinate system describes how an image was acquired with respect to the anatomy. Medical scanners create regular, rectangular arrays of points and cells which start at the upper left corner. The x axis increases to the right, the y axis to the bottom and the z axis backwards. In addition to the intensity value of each voxel (x y z) the origin and spacing of the anatomical coordinates are stored too. You can also think of the system as indexes of a 3D array.
* The origin represents the position of the first voxel (0,0,0) in the anatomical coordinate system, e.g. (100mm, 50mm, -25mm)
* The spacing specifies the distance between voxels along each axis, e.g. (1.5mm, 0.5mm, 0.5mm)

The code supports both coordinate systems, you just need to indicate which one you are using when supplying coordinates (in [config.py](config.py) and via the function call itself).


## Using the Code
To configure data load/save locations and other system parameters, open the [config.py](config.py) file and change the contents accordingly. The settings you can change are as follows
```
## Data Location ##
'healthy_scans_raw'     #path to directory where the healthy scans are. Filename is patient ID.
'healthy_coords'        #path to csv where each row indicates where a healthy sample is (format: filename, x, y, z). 
                        #    'filename' is the folder containing the dcm files of that scan or the mhd file name, slice is the z axis
'healthy_samples'       #path to pickle dump of processed healthy samples for training.
'unhealthy_scans_raw'   #path to directory where the unhealthy scans are
'unhealthy_coords'      #path to csv where each row indicates where a healthy sample is (format: filename, x, y ,z)
'unhealthy_samples'     #path to pickle dump of processed healthy samples for training.
'traindata_coordSystem' #the coord system used to note the locations of the evidence in the training data scans ('world' or 'vox'). 

## Model & Progress Location ##
'modelpath_inject'      #path to save/load trained models and normalization parameters for injector
'modelpath_remove'      #path to save/load trained models and normalization parameters for remover
'progress' = "images"  #path to save snapshots of training progress

## tensorflow configuration ##
'gpus' #sets which GPU to use (use_CPU:"", use_GPU0:"0", etc...)

## CT-GAN Configuration ##
'cube_shape'            #dimensions (z,y,x) of the cuboid to be cut out of the scan. Default is 32x32x32.
'mask_xlims'            #length of the in-painting mask on the x-axis, centered within the cuboid.     
'mask_ylims'            #... 
'mask_zlims'            #...
'copynoise'             #indicates if the noise touch-up is copied onto the tampered region from a hardcoded coordinate, 
                        #    or if Gaussain interpolated noise should be used
```
We will now review how to use the example scripts provided in this repo.

### Step 1: Build a Trainingset
To build a training set for injecting evidence, you need positive examples. To extract this dataset from a set of medical scans, run
```
$ python 1A_build_injector_trainset.py
```
This code will load the medical scans indicated in [config.py](config.py) and create a scaled, equalized, and normalized instance from each coordinate listed in the csv 'unhealthy_coords'. Each instance is then augmented into 66 alternate versions via shifts, flips, and rotations.
The final set of instances are then saved as a numpy serialized data file in the file 'unhealthy_samples'. 
To build a training set for removing evidence, run
```
$ python 1B_build_remover_trainset.py
```
Once extraction is complete you will be shown a plot containing some samples:

*Example plot shown after extraction showing random instances extracted (middle slice only):*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/extractor_fig.png)

### Step 2: Train the cGANs
To train a cGAN capable of injecting/removing evidence, run
```
$ python 2A_train_injector.py
```
and/or
```
$ python 2B_train_remover.py
```
This code will use the preprocessed dataset you have created in step 1A/1B and the setting in [config.py](config.py) to train the generator models.
Snapshots of the progress are saved to a local ‘images’ directory in png format (default is after every 50 batches). For example:

*Example progress snapshot after 50 batches while training the injector:*
![](https://raw.githubusercontent.com/ymirsky/CT-GAN/master/readme/0_50.png)

 
### Step 3: Tamper Medical Imagery
To inject evidence into a scan using the trained models, run
```
$ python 3A_inject_evidence.py
```
or
```
$ python 3B_remove_evidence.py	
```
This code will load a scan, inject/remove evidence at the provided coordinates, and then save the tampered scan. Here, the load/save locations along with the target coordinates must be changed within the example script (not via config.py). Be sure to note which coordinate system you are using.

For easier use, there is an interactive GUI which can be used to scroll through scans and tamper them by clicking on the Image. The results can also be saved in dicom format. To use this tool, run
```
$python GUI.py load_dir save_dir
```
If the load directory is not supplied then the data/healthy_scans directory is used. The save directory is data/tampered_scans unless specified.

There is also command-line tool 'tamper.py' which can be used to inject/remove evidence using your trained models. For help using this tool, run
```
$python tamper.py -h

usage: tamper.py [-h] -t TARGET -d DESTINATION -a {inject,remove} -c
                 [COORD [COORD ...]] [-s {vox,world}] [-f {dicom,numpy}]

CT-GAN medical evidence tamper pipeline. Uses a pre-trained injection/removal model to tamper evidence in a given 3D medical scan (mhd/raw or dicom series)

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        The path to the target scan to be tampered. The path should point to the directory containing a dicom series (*.dcm files) or a *.mhd file.
  -d DESTINATION, --destination DESTINATION
                        The directory (path) to save the tampered scan.
  -a {inject,remove}, --action {inject,remove}
                        The directory (path) to save the tampered scan.
  -c [COORD [COORD ...]], --coord [COORD [COORD ...]]
                        The selected coordinate(s) in the target scan to inject evidence. You must provide one or more coordinates in z,y,x format with no spaces.
                        Example (inject at two locations): python tamper.py -t patient.mhd -d outdir -c 123,324,401 53,201,441
  -s {vox,world}, --system {vox,world}
                        Indicate the coordinate system of the supplied target coordinates: 'vox' or 'world'. Default is 'vox'.
  -f {dicom,numpy}, --outformat {dicom,numpy}
                        The output format to save the tamepred scan: 'dicom' or 'numpy'. Default is 'dicom'.

To change other settings, check config.py
For more information, please read our paper:
CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning.
Yisroel Mirsky, Tom Mahler, Ilan Shelef, and Yuval Elovici

```

# Datasets
In our research we investigated injection and removal of cancerous lung nodules in CT scans. The scans were obtained from the Cancer Imaging Archive [found here](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

In [this link](https://drive.google.com/open?id=1R0WD_5IZ3NlyCiOPf1Ex74nBnZYQegwr) you will find the *tampered* scans which we used in our blind and open experiments. These scans may be helpful if you are researching means for detecting these types of attacks. We also supply labels and our results from our radiologists and the AI.

# License
See the [LICENSE](LICENSE) file for details


# Citations
If you use the source code in any way, please cite:

*Yisroel Mirsky, Tom Mahler, Ilan Shelef, and Yuval Elovici. 28th USENIX Security Symposium (USENIX Security 19)*
```
@inproceedings {236284,
author = {Yisroel Mirsky and Tom Mahler and Ilan Shelef and Yuval Elovici},
title = {CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning},
booktitle = {28th {USENIX} Security Symposium ({USENIX} Security 19)},
year = {2019},
isbn = {978-1-939133-06-9},
address = {Santa Clara, CA},
pages = {461--478},
url = {https://www.usenix.org/conference/usenixsecurity19/presentation/mirsky},
publisher = {{USENIX} Association},
month = aug,
}
```

Yisroel Mirsky
yisroel@post.bgu.ac.il
