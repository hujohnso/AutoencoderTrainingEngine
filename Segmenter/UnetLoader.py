from keras.optimizers import Adam

from Segmenter.zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef


class UnetLoader:
    model = None

    def __init__(self):
        model = ZF_UNET_224(weights='generator')
        optim = Adam()
        model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
        model.load_weights("PreTrainedUnet/zf_unet_224.h5")

    def load_unet(self):
        return self.model