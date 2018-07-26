# TF_Convolutional_Autoencoder
### Convolutional autoencoder for encoding/decoding RGB images in TensorFlow

This is a sample template adapted from Arash Saber Tehrani's Deep-Convolutional-AutoEncoder tutorial https://github.com/arashsaber/Deep-Convolutional-AutoEncoder for 3-channel images. I have tested this implementation on rescaled samples from the CelebA dataset from CUHK http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html to produce reasonably decent results.

Both inputs to the encoder and outputs from the decoder are available for viewing in TensorBoard.

## Outputs

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

