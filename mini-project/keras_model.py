from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape, Dropout, SpatialDropout2D
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import glob


train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=150
validation_samples=807

save_figures = True
SAVE_PATH = '/home/grupo02/mariaw5/3layers_k3_mp_bn_adam0.0001_rlrop_gap_lessda/'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


model = Sequential()
model.add(Conv2D(64, kernel_size=3,activation = 'relu', input_shape= (img_width, img_height, 3)))
#model.add(Conv2D(32, kernel_size=3,activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.5))
model.add(Conv2D(32, (1,1), activation='relu'))
model.add(Conv2D(32, kernel_size=3,activation = 'relu'))
#model.add(Conv2D(32, kernel_size=3,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
#model.add(SpatialDropout2D(0.5))
model.add(Conv2D(16, (1,1), activation='relu'))
#model.add(Conv2D(16, kernel_size=3,activation = 'relu'))
model.add(Conv2D(32, kernel_size=3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=3, activation='relu'))
#model.add(SpatialDropout2D(0.5))
#model.add(Conv2D(16, kernel_size=3, activation='relu'))
#model.add(Conv2D(16, kernel_size=3, activation='relu'))
#for layer_size in layers[1:]:
#	model.add(Conv2D(layer_size, kernel_size=3, activation = 'relu'))
#	model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
#	model.add(BatchNormalization())
#model.add(Flatten())
#model.add(BatchNormalization())
model.add(GlobalAveragePooling2D(data_format=None))
model.add(BatchNormalization())
model.add(Dense(8, activation = 'softmax'))


#optim = optimizers.SGD(lr=0.1, momentum=0.1, nesterov=True)
#optim = optimizers.Adamax()
#optim = optimizers.Adadelta()
optim = optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optim, metrics=['accuracy'])
plot_model(model, to_file=str(layers)+'modelMLP.png', show_shapes=True, show_layer_names=True)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
	horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_data_dir,
 	target_size=(img_width, img_height),
 	batch_size=batch_size,
 	class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
 	target_size=(img_width, img_height),
 	batch_size=batch_size,
 	class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
 	target_size=(img_width, img_height),
 	batch_size=batch_size,
 	class_mode='categorical')

history=model.fit_generator(train_generator,
 	steps_per_epoch=(int(1881//batch_size)+1),
 	nb_epoch=number_of_epoch,
 	validation_data=validation_generator,
 	validation_steps= (int(validation_samples//batch_size)+1),
	callbacks=[reduce_lr])

print(model.summary())
print("Accuracy history of layer: "+str(layers)+ "\n")
print(str(history.history['val_acc'])+"\n")


result = model.evaluate_generator(test_generator, val_samples=validation_samples)
print("RESULT: \n")
print(result)


# list all data in history

if save_figures:
# summarize history for accuracy
	plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(SAVE_PATH+str(layers)+'accuracy.jpg')
	plt.close()
# summarize history for loss
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(SAVE_PATH+str(layers)+'loss.jpg')
