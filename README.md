# TF_Convolutional_Autoencoder
#### _Convolutional autoencoder for encoding/decoding RGB images in TensorFlow_

This is a sample template adapted from Arash Saber Tehrani's Deep-Convolutional-AutoEncoder tutorial https://github.com/arashsaber/Deep-Convolutional-AutoEncoder for encoding/decoding 3-channel images. The template has been fully commented. I have tested this implementation on rescaled samples from the CelebA dataset from CUHK http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html to produce reasonably decent results.

**Add on features:**
* Takes 3-channel images as input instead of MNIST
* Training now performs checkpoint saves and restores
* Both inputs to the encoder and outputs from the decoder are available for viewing in TensorBoard
* _Input autorescaling (not yet functional)_
* ReLU activation replaced by LeakyReLU to resolve dying ReLU

**Caveats:**
* It is highly recommended to perform training on the GPU (Took ~2 hrs to train 500 epochs on a Tesla K80 for CelebF).
* The input size can be increased, but during testing OOM errors occured on the K80 for the input size of 84x84. This will be fixed in a later update. For now if you get any OOM errors in tensor allocation, try to reduce the input size.
* Output is currently visibly undersaturated.

## Outputs
N.B. The input image is only 42x42 which is why it looks blurry.

## How to run
1. Make sure to create directory `./logs/run1/` to save TensorBoard output. For pushing multiple runs to TensorBoard, simply save additional logs as `./logs/run2/`, `./logs/run3/` etc.

2. Unzip `./CelebF.tar.gz` and save jpegs in ./data/CelebF/

3. Either use provided image set or your own. If using your own dataset, I recommend ImageMagick for resizing: https://www.imagemagick.org/script/download.php

4. If using ImageMagick, start Bash in `./data/<your_dir>/`:
```
for file in $PWD/*.jpg
do
convert $file -resize 42x42 $file
done
```

5. `python ConvAutoencoder.py`

