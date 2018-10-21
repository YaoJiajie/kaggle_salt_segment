# Kaggle TGS Salt Identification Challenge (Top 6% model)

## Model

- UNet based on `RestNet34`
- With `SpatialSqueeze` and `ChannelSqueeze` 
- Using `BCE loss` & `LovaszSoftmax loss`

## Dependencies

- Pytorch 0.4.1
- Cuda 9.1

## Train & Test

1. Download data:

    $ kaggle competitions download -c tgs-salt-identification-challenge
   
2. Convert data:

    $ python3 data.py # by default, create 6-folders, with augmentation
   
3. Training

    $ python3 train.py 1  weights_folder_1  # start train on folder-1, and save weights to weights_folder_1
    
4. Check the training status using `plot_logs.py`.

5. Run tests & create submitting files using `test.py`
