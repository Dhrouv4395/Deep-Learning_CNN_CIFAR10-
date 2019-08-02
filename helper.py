import pickle  #to decode files
import numpy as np  
from keras.utils import np_utils #to make one-hot vectors
import matplotlib.pyplot as plt #to plot graphs

#constants
path= 'data_p/'

#height or width of the image
size = 32

#3 channels RGB
channels = 3

#number of classes
num_classes = 10

#each file contains 10,000 images
image_batch = 10000

# 5 training files
num_files_train = 5

#total number of training images
images_train = image_batch * num_files_train

def unpickle(file):
    #convert byte stream to object
    with open(path + file,'rb') as fo:
        print('Decoding file: %s'%(path+file))
        dict = pickle.load(fo,encoding='bytes')
    #dictionary with images and labels
    return dict

def convert_images(raw_images):
    #convert images to numpy arrays
    #convert raw images to numpy images and normalize it
    raw = np.array(raw_images, dtype=float) / 255.0

    #Reshape to 4-dimensons[image_number,channel,height,width]
    images = raw.reshape([-1, channels, size, size])

    images = images.transpose([0,2,3,1])

    #4D array[image_number,height,width,channel]

    return images

def load_data(file):
    #load file, unpicle it and return images with their labels
    data = unpickle(file)
    #get raw images
    images_array = data[b'data']
    #convert images
    images = convert_images(images_array)
    #convert class number to numpy array
    labels = np.array(data[b'labels'])
    #images and labels in np array form
    return images, labels

def get_test_data():
    #load all test data
    images,labels = load_data(file='test_batch')
    #images, there labels and
    #corresponding one-hot vectors in form of np arrays
    return images,labels,np_utils.to_categorical(labels,num_classes)

def get_train_data():
    #load all training data n 5 files
    #pre allocate arrays
    images = np.zeros(shape=[images_train,size,size,channels],dtype=float)
    labels = np.zeros(shape=[images_train],dtype=float)
    #starting index of training dataset
    start = 0
    #for all 5 files
    for i in range(num_files_train):
        #load images and labels
        images_batch, labels_batch = load_data(file = 'data_batch'+str(i+1))
        #calculagte end index for current batch
        end = start + image_batch
        #store data in  corresponding arrays
        images[start:end,:] = images_batch
        labels[start:end] = labels_batch
        #update starting index of next batch
        start = end

    #Images,theres labels and
    #corresponding one-hot vectors in form of np arrays
    return images, labels, np_utils.to_categorical(labels,num_classes)

def get_class_names():
    #load class names
    raw = unpickle('batches.meta')[b'label_names']
    #covert from binary strings
    names = [x.decode('utf-8') for x in raw]
    #class names
    return names

def plot_images(images,labels_true,class_names,labels_pred=None):
    assert len(images) == len(labels_true)
    #create a figure with sub-plots
    fig,axes  = plt.subplot(3, 3, figsize = (8, 8))
    #Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace,wspace=0.3)

    for i, ax in enumerate(axes.flat):
        #fix crash when less than 9 images
        if i < len(images):
            #plot the images
            ax.imshow(images[i],interpolation='spline16')
            #name of the true class
            labels_true_name = class_name[labels_true[i]]
            #show true and predicted classes
            if labels_pred is None:
                xlabel='True: ' + labels_true_name
            else:
                #Name o the predicted classes
                labels_pred_name = class_name[labels_pred[i]]
                xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name
            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()

def plot_model(model_details):
    # Create sub-plots
    fig, axs = plt.subplots(1, 2, figsize = (15, 5))
    
    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
    axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc = 'best')
    
    # Summarize history for loss
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
    # Show the plot
    plt.show()

def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
    
    incorrect = (correct == False)
    
    # Images of the test-set that have been incorrectly classified.
    images_error = images_test[incorrect]
    
    # Get predicted classes for those images
    labels_error = labels_pred[incorrect]

    # Get true classes for those images
    labels_true = labels_test[incorrect]
        
    # Plot the first 9 images.
    plot_images(images=images_error[0:9],
                labels_true=labels_true[0:9],
                class_names=class_names,
                labels_pred=labels_error[0:9])
    
    
def predict_classes(model, images_test, labels_test):
    
    # Predict class of image using model
    class_pred = model.predict(images_test, batch_size=32)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == labels_test)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred
