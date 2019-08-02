import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Lambda,Conv2D,MaxPooling2D,Dropout,Dense,Flatten,Activation

from helper import get_class_names,get_train_data,get_test_data,plot_images,plot_model

matplotlib.style.use('ggplot')

#import class names
class_names = get_class_names()
#print(class_names)
num_classes = len(class_names)
#print(num_classes)

#height and width of the images
IMAGE_SIZE=32
#3 channels RGB
CHANNELS = 3

#Fetch and decode Data
#Load the training dataset, labels are integers whereas class is one-hot vectors

images_train, labels_train, class_train = get_train_data()
#print('Labels_Train: ',labels_train)
#print('Class_Train: ',class_train)

images_test, labels_test, class_test = get_test_data()
#print('Training set size:\t',len(images_train))
#print('Testing set size:\t',len(images_test))

def cnn_model():
    model = Sequential()
    
    model.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.save('cifar10_cnn_model')

    model.summary()

    return model

model = cnn_model()
#print(model)

checkpoint = ModelCheckpoint('best_model_simple.h5', #model filename
                            monitor='val_loss', #quantity to monitor
                            verbose=0,#verbosity = 0 or 1
                            save_best_only=True,#The latest best model will not be overwritten
                            mode='auto')#The decision to overwrite model is made)
                                    #automatically depending on the quantity to monitor


model.compile(loss='categorical_crossentropy',#Better loss function for neural networs
            optimizer=Adam(lr=1.0e-4),#Adam optimizer with 1.0e-4 learning rate
            metrics=['accuracy'])#Metrics to be evaluted by model

model_details = model.fit(images_train,class_train,
                          batch_size=128,#number of sample per gradient update
                          epochs=10,#number of iterations
                          validation_data=(images_test,class_test),
                          callbacks=[checkpoint],
                          verbose=1)
#print(model_details)

scores = model.evaluate(images_test,class_test,verbose=0)
#print('Accuracy: %.2f%%'%(scores[1]*100))

#plot_model(model_details)

class_pred = model.predict(images_test,batch_size=32)
#print(class_pred[0])

labels_pred = np.argmax(class_pred,axis=1)
#print(labels_pred)

correct = (labels_pred==labels_test)
#print(correct)
#print('Number of currect prediction: %d'%sum(correct))

num_images = len(correct)
#print('Accurecy: %.2f%%'%((sum(currect)*100)/num_images))

incorrect = (correct == False)
#Images of the test-set that have been incorrectly classified.
images_error = images_test[incorrect]
#Get predicted classes for those images
labels_error = labels_pred[incorrect]
#Get true classes for those images.
labels_true = labels_test[incorrect]

prediction = model.predict(images_test)

def plot_images(i,prediction_array,true_label,img):
	prediction_array, true_label, img = prediction_array[i], true_label[i], img[i]
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap = plt.cm.binary)
	predicted_label = np.argmax(prediction_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
				    100*np.max(prediction_array),
				    class_names[true_label],
                                    color = color))

def plot_value_array(i,prediction_array,true_label):
	prediction_array, true_label = prediction_array[i], true_label[i]
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), prediction_array, color = '#777777' )
	plt.ylim([0,1])
	predicted_label = np.argmax(prediction_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, num_cols*2, 2*i+1)
	plot_images(i,prediction,labels_test, images_test)
	plt.subplot(num_rows,num_cols*2, 2*i+2)
	plot_value_array(i,prediction,labels_test)
plt.show()
    

'''plot_images(images=images_error[0:9],
            labels_true=labels_true[0:9],
            class_names=class_names,
            labels_pred=labels_error[0:9])'''
