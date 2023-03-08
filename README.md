## Earth Surface Water Image Segmentation

We use satellite images from Sentinel 2 to identify which portions of the image correspond to water, using the dataset from [Luo et al.'s 2021 paper](https://www.sciencedirect.com/science/article/pii/S0303243421001793). 

For semantic segmentation, we use the U-Net architecture, with a Resnet50 backbone for our encoder. To this end, we only keep the R, G, and B bands of the 13 bands in Sentinel 2 (actually, Luo et al. already threw away 7 bands when making their dataset.)
By keeping the RGB bands only, I can then use pre-trained ImageNet weights.

For training, we first freeze the pre-trained layers and optimize the decoder layers. Afterwards, we unfreeze the pre-trained layers and fine-tune the all layers.

This was all done in Pytorch, and was especially made convenient with TorchGeo.
