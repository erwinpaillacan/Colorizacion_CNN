import os
from data_utils_CNN import train_generator, val_batch_generator
from model_utils_CNN import generar_modelo
from train_utils import TensorBoardBatch
from keras.callbacks import ModelCheckpoint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
nb_train_images = 128 # con cuantas imagenes se quiere entrenar
batch_size = 16 #se toman de poco
epocas= 5000 #cuantas veces se pasa cada batch por la red
model = generar_modelo(lr=1e-4)
model.summary()


# use Batchwise TensorBoard callback
tensorboard = TensorBoardBatch(batch_size=batch_size)
checkpoint = ModelCheckpoint('weights/pesos_CNN_puro_mse.h5', monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True)
callbacks = [checkpoint, tensorboard]

gen=train_generator(batch_size)
val=val_batch_generator(batch_size)
model.fit_generator(gen,
                    steps_per_epoch=nb_train_images // batch_size,
                    epochs=epocas,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val,
                    validation_steps=1
                    )
