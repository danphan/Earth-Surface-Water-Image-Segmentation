from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
from tqdm import tqdm
import rioxarray as rio
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_setup import create_dataloader

MAX_PIXEL_VAL = 65535.0 #because satellite imagery is stored as uint16's
WIDTH = 512
HEIGHT = 512
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
T_MAX = 5
MODEL_PATH = 'models/unet.pt'


root = Path('data/dset-s2')

train_imgs = list((root/'tra_scene').glob('*tif'))
train_masks = list((root/'tra_truth').glob('*tif'))

val_imgs = list((root/'val_scene').glob('*.tif'))
val_masks = list((root/'val_truth').glob('*.tif'))


train_dataloader = create_dataloader(train_imgs,
                                     train_masks,
                                     img_size = (WIDTH, HEIGHT),
                                     batch_size = BATCH_SIZE,
                                     train = True)
val_dataloader = create_dataloader(val_imgs,
                                   val_masks,
                                   img_size = (WIDTH, HEIGHT),
                                   batch_size = BATCH_SIZE,
                                   train = False)

def training_loop(epochs, train_dl, val_dl, model, loss_fn, optimizer, scheduler = None):

    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    iou_metric = JaccardIndex(task = 'multiclass', num_classes = 2)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        accum_loss = 0
        iou = 0
        for batch in tqdm(train_dl):
            X = batch['image']
            y = batch['mask'].squeeze(dim = 1).type(torch.long)
            pred = model(X)
            loss = loss_fn(pred, y)

            #update loss
            accum_loss += float(loss) / len(train_dl)
            iou += iou_metric(preds = pred, target = y)/len(train_dl)

            #Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'train loss: {accum_loss:.5f}')
        print(f'train IOU: {iou:.5f}')

        train_losses.append(accum_loss)
        train_ious.append(float(iou))

        if val_dl is not None:
            print('Calculating validation metrics')
            val_loss = 0
            iou = 0
            with torch.no_grad():
                for batch in tqdm(val_dl):
                    X = batch['image']
                    y = batch['mask'].squeeze(dim = 1).type(torch.long)
                    pred = model(X)
                    val_loss += loss_fn(pred, y)/len(val_dl) #update avg loss across batches
                    iou += iou_metric(preds = pred, target = y)/len(val_dl)

            print(f'val loss: {val_loss:.5f}')
            print(f'val IOU: {iou:.5f}')
            val_losses.append(float(val_loss))
            val_ious.append(float(iou))

        if scheduler is not None:
            scheduler.step()

    if val_dl:
        return {'train_loss' : train_losses,
                'val_loss' : val_losses,
                'train_iou' : train_ious,
                'val_iou' : val_ious}

    return {'train_loss' : train_losses,
            'train_iou' : train_ious}

def evaluate(model, loss_fn, val_dl):

    iou_metric = JaccardIndex(task = 'multiclass', num_classes = 2)

    val_loss = 0
    iou = 0

    print('Calculating validation metrics')
    with torch.no_grad():
        for batch in tqdm(val_dl):
            X = batch['image']
            y = batch['mask'].squeeze(dim = 1).type(torch.long)
            pred = model(X)
            val_loss += loss_fn(pred, y)/len(val_dl) #update avg loss across batches
            iou += iou_metric(preds = pred, target = y)/len(val_dl)

    print(f'val loss: {val_loss:.5f}')
    print(f'val IOU: {iou:.5f}')

    return {'val_loss' : val_loss,
            'val_iou' : iou}

#define model
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    in_channels=6,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)


if Path(MODEL_PATH).exists():
    print('loading model')
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

loss_fn = smp.losses.DiceLoss(mode = 'multiclass')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)
scheduler = CosineAnnealingLR(optimizer, T_max = T_MAX)
train_dict = training_loop(NUM_EPOCHS, train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)

#save model parameters
torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
    #plot metrics vs epoch
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
    ax[0].plot(np.arange(NUM_EPOCHS),train_dict['train_loss'],'o-', label = 'train')
    ax[0].plot(np.arange(NUM_EPOCHS),train_dict['val_loss'],'o-', label = 'val')
    ax[1].plot(np.arange(NUM_EPOCHS),train_dict['train_iou'],'o-', label = 'train')
    ax[1].plot(np.arange(NUM_EPOCHS),train_dict['val_iou'],'o-', label = 'val')
    ax[0].set_title('loss')
    ax[1].set_title('IOU')
    plt.show()
    
