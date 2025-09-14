
part-of-cars - v9 2023-04-10 10:50pm
==============================

This dataset was exported via roboflow.com on April 11, 2023 at 4:51 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2477 images.
Part-of-cars are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Resize to 256x256 (Stretch)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -26 and +26 degrees
* Random Gaussian blur of between 0 and 1 pixels
* Salt and pepper noise was applied to 1 percent of pixels


