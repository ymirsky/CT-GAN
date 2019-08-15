from procedures.trainer import *

print("Training CT-GAN Injector...")
CTGAN_inj = Trainer(isInjector = True)
CTGAN_inj.train(epochs=200, batch_size=32, sample_interval=50)
print('Done.')