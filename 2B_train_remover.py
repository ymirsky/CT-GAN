from procedures.trainer import *

print("Training CT-GAN Remover...")
CTGAN_rem = Trainer(isInjector = False)
CTGAN_rem.train(epochs=200, batch_size=32, sample_interval=50)
print('Done.')