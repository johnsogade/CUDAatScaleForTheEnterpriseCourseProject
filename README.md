# CUDAatScaleForTheEnterpriseCourseProject
This is for the course project for the CUDA at Scale for the Enterprise

## Project Description

This project was created as a casual way for investigating some of the image and signal processing capabilities of CUDA NPP (Nvidia's Performance Primitive) library and requires at least C++17 to compile.
Currently, the project provides an implementation for the following image processing filters:

		•	Filter Box with Border Control (nppiFilterBoxBorder_8u_C1R, nppiFilterBoxBorder_8u_C3R, etc.)
		•	Gauss Filter with Border Control (nppiFilterGaussBorder_8u_C1R, nppiFilterGaussBorder_8u_C3R, etc.)
		
The project allows images placed in the 'data/' folder to be processed based on filter specifications entered in the 'run.sh' script file. The specifications include filter type, the filename or directory of the output file, the mask size, source offset, and/or anchor. Image processing is done by running the ‘run.sh’ script with or without arguments. If started without arguments, ‘run.sh’ will process all images placed in the ‘data/’ folder. An example argument is "-input=../data/sloth.png -maskSize=25 -filter=1", where ‘-input’ specifies the input file path, “-maskSize” the mask size, and “-filter” the filter type (“1” for box and “2” for gauss filter). Other possible argument settings are “srcOffset” the source offset, and “anchor” the anchor. Note that the single value for mask size is used for both dimensions i.e. both width and height are same (25, 25), and the same applies to the source offset and anchor as well. Currently, the supported image formats are those that can be loaded by the freeware software, 'FreeImage’, used by the Nvidia NPP samples. Tests were conducted on about 150 regular images in the BMP, PNG, PGM, TIFF formats that span 1 (gray-scale), 3, or 4 channels. The 2-channel image filter was not implemented.  Each processed image is entered into a log file and the files dataFilterRecord_box.log, and dataFilterRecord_gauss.log were created during the 150-image test run (for Box and Gauss filter respectively). The file dataFilterRecord_sample.log was created during a test run using the images currently available in the ‘data/’ folder.
It's easy to see the effect of the filter by comparing the image “sloth.png” to “sloth_boxFilter.png” or the image “building1.png” to “building1_gaussFilter.png”.
One of the observations made during testing is that multiple runs of the NPP kernel during a single program instance fails; therefore, the only way to run multiple images is using the script “run.sh” which invokes the executable for each processed image. The code to do multiple NPP kernel runs in a single program instance is however available in the project albeit unused. Another obervation is that for border control ony the type "NPP_BORDER_REPLICATE" was tested as the Nvidia NPP documentation states somewhere that it's the only currently supported border type.
The code is structured in such a way as to easily extend to handle testing of other features of NPP. 

The project is structured following the form here:

https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate
 

## Code Organization

```bin/```
This folder holds all binary/executable code that is built using the 'build.sh' script. Executable code should use the .exe extension or Operating System-specific extension.

```data/```
This folder should hold all example data in any format. It currently has a small sample of test images in the 'data/' folder processed into the "boxFilter" or "gaussFilter" subfolders.
respectively (based on either the use of the Box or Gauss filter). If the original data is rather large or can be brought in via scripts, this can be left blank in the repository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code is placed here. 

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible, it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile and/or build.sh```
The Makefile script is placed in the 'scr/' folder and it is adapted from the one provided with the Nvidia cuda sample. It should be used in conjunctionwith the 'build.sh' script for building the project's code in an automatic fashion.

```run.sh```
An optional script used to run the executable code, either with or without command-line arguments. If the command line arguments is not provided, this script will process all the images in the 'data/' folder based on filter specifications in the 'run.sh' script.
