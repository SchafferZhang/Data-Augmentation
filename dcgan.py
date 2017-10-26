import matplotlib
matplotlib.use('Agg')
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
import argparse
import math
import matplotlib.pyplot as plt


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28,1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    # global X_train,y_train,X_test,y_test
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.concatenate([X_train[y_train==0],X_train[y_train==8]],axis=0)
    y_train = np.concatenate([y_train[y_train==0],y_train[y_train==8]-7],axis=0)
    ind = np.random.permutation(len(y_train))
    X_train = X_train[ind]
    y_train = y_train[ind]
    y_train = keras.utils.to_categorical(y_train, 4)
    
    # y_test = keras.utils.to_categorical(y_test, 2)
    X_train = (X_train.astype(np.float32) - 127.5)/128.
    X_train = X_train[:,:,:,np.newaxis]

    X_test = np.concatenate([X_test[y_test==0],X_test[y_test==8]],axis=0)
    y_test = np.concatenate([y_test[y_test==0],y_test[y_test==8]-7],axis=0)
    X_test = (X_test.astype(np.float32) - 127.5)/128.
    X_test = X_test[:,:,:,np.newaxis]
    
    print 'The training data size is: ',X_train.shape[0]
    print 'The testing data size is: ',X_test.shape[0]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    datagen = ImageDataGenerator(featurewise_center=False,
                                                           samplewise_center=False,
                                                           featurewise_std_normalization=False,
                                                           samplewise_std_normalization=False,
                                                           zca_whitening=False,
                                                           zca_epsilon=1e-6,
                                                           rotation_range=30.,
                                                           width_shift_range=0.1,
                                                           height_shift_range=0.1,
                                                           shear_range=0.,
                                                           zoom_range=0.,
                                                           channel_shift_range=0.,
                                                           fill_mode='nearest',
                                                           cval=0.,
                                                           horizontal_flip=True,
                                                           vertical_flip=False,
                                                           rescale=None,
                                                           preprocessing_function=None,
                                                           data_format=None)

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='categorical_crossentropy', optimizer=g_optim,metrics=['acc'])
    d.trainable = True
    d.compile(loss='categorical_crossentropy', optimizer=d_optim,metrics=['acc'])
    d_list = []
    g_list = []
    accuracy = []
    for epoch in range(300):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        index = 0
        for x_batch, y_batch in datagen.flow(X_train, y_train,  batch_size=BATCH_SIZE):
            index += 1
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 96))
            cat = np.random.choice([2,3], BATCH_SIZE,p=[0.5,0.5])
            cat_one_hot = keras.utils.to_categorical(cat, 4)
            noise = np.concatenate([noise,cat_one_hot],axis=1)
            # image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # y_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 50 == 0:
                image = combine_images(generated_images)
                image = image*128+127.5
                Image.fromarray(image.astype(np.uint8)).save(os.path.join('samples',
                    str(epoch)+"_"+str(index)+".png"))
            X = np.concatenate((x_batch, generated_images))
            # y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            y = np.concatenate((y_batch,cat_one_hot),axis=0)
            d_loss = d.train_on_batch(X, y)
            d_list.append(d_loss[1]) #measure the performance of the D
            print("batch %d d_loss : %f" % (index, d_loss[0]))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE*2, 96))
            cat = np.random.choice([2,3], BATCH_SIZE*2,p=[0.5,0.5])
            cat_one_hot = keras.utils.to_categorical(cat, 4)
            noise = np.concatenate((noise,cat_one_hot),axis = 1)
            y_fake = keras.utils.to_categorical(cat-2, 4)

            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, y_fake)
            g_list.append(g_loss[1]) #measure the performance of the G
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss[0]))

            if index >= len(X_train)/BATCH_SIZE:
                break
            
        g.save_weights(os.path.join('weights','generator_'+str(epoch)), True)
        d.save_weights(os.path.join('weights','discriminator_'+str(epoch)), True)
        num_batches = X_test.shape[0]/BATCH_SIZE
        acc = 0.
        for i in range(num_batches):
            X_batch = X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            pred = d.predict_on_batch(X_batch)
            # print(pred.shape)
            pred_out = np.ndarray(shape=(BATCH_SIZE,num_classes),dtype=np.float32)
            # print(pred_out.shape)
            for i in range(num_classes):
                pred_out[:,i] = pred[:,i] + pred[:,i+num_classes]
            pred_cls = np.squeeze(pred_out.argmax(axis=1))
            # print(pred_cls)
            # print(y_batch)
                # assert pred_cls.shape == 100
            acc += np.mean(np.cast[np.float32](pred_cls==y_batch))
            # print(acc)
        acc = acc/num_batches
        accuracy.append(acc)
        print('====================================================')
        print('                         The validation accuracy is %f'%acc)
        print('====================================================')
        
    print('================================')
    print('The index of the highest acc is at epoch %d, which is %f'%(accuracy.index(max(accuracy)),max(accuracy)))
    print('================================')

    plt.plot(d_list[::len(X_train)/BATCH_SIZE])
    plt.plot(g_list[::len(X_train)/BATCH_SIZE])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['discriminator','generator'],loc='upper left')
    plt.savefig(os.path.join('fig','GAN-augmentation','model-accuracy.png'))
    plt.clf()
    plt.plot(accuracy)
    plt.title('validation accuray')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join('fig','GAN-augmentation','classification-accuracy.png'))
    plt.clf()

def test(BATCH_SIZE):
    d = discriminator_model()
    d.compile(loss='categorical_crossentropy',optimizer='SGD')
    d.load_weights(os.path.join('weights','discriminator_277'))
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = np.concatenate([X_test[y_test==0],X_test[y_test==8]],axis=0)
    y_test = np.concatenate([y_test[y_test==0],y_test[y_test==8]-7],axis=0)
    X_test = (X_test.astype(np.float32) - 127.5)/128.
    X_test = X_test[:,:,:,np.newaxis]
    print(X_test.shape[0])
    num_batches = X_test.shape[0]/BATCH_SIZE
    acc = 0.
    for i in range(num_batches):
        X_batch = X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        pred = d.predict_on_batch(X_batch)
        # print(pred.shape)
        pred_out = np.ndarray(shape=(BATCH_SIZE,num_classes),dtype=np.float32)
        # print(pred_out.shape)
        for i in range(num_classes):
            pred_out[:,i] = pred[:,i] + pred[:,i+num_classes]
        pred_cls = np.squeeze(pred_out.argmax(axis=1))
        # print(pred_cls)
        # print(y_batch)
            # assert pred_cls.shape == 100
        acc += np.mean(np.cast[np.float32](pred_cls==y_batch))
        # print(acc)
    acc = acc/num_batches
    print('====================================================')
    print('                         The validation accuracy is %f'%acc)
    print('====================================================')
    return acc



def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='categorical_crossentropy', optimizer="SGD")
    g.load_weights(os.path.join('weights','generator_277'))
    if nice:
        d = discriminator_model()
        d.compile(loss='categorical_crossentropy', optimizer="SGD")
        d.load_weights(os.path.join('weights','discriminator_277'))
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 96))
        # cat = np.random.choice([2,3], BATCH_SIZE*20,p=[0.5,0.5])
        cat = [3]*BATCH_SIZE*20
        cat_one_hot = keras.utils.to_categorical(cat, 4)
        noise = np.concatenate((noise,cat_one_hot),axis=1)
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)[:,1]
        # d_pret.resize(-1,1)
        d_pret = np.reshape(d_pret, (BATCH_SIZE*20,1))
        # print(d_pret.shape)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 96))
        cat = [3]*BATCH_SIZE
        cat_one_hot = keras.utils.to_categorical(cat, 4)
        noise = np.concatenate((noise,cat_one_hot),axis=1)
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*128+127.5
    Image.fromarray(image.astype(np.uint8)).save(os.path.join('generation',
        "generated_image.png"))
def train_classifier(BATCH_SIZE):
    model = Sequential()
    model.add(
                        Conv2D(64,(5,5),
                        padding='same',
                        input_shape=(28,28,1))
                        )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(5,5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    optim = SGD(lr=0.0005, momentum=0.9,nesterov=True)
    model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['acc'])
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.concatenate([X_train[y_train==0],X_train[y_train==8]],axis=0)
    y_train = np.concatenate([y_train[y_train==0],y_train[y_train==8]-7],axis=0)
    ind = np.random.permutation(len(y_train))
    X_train = X_train[ind]
    y_train = y_train[ind]
    y_train = keras.utils.to_categorical(y_train, 2)
    
    # y_test = keras.utils.to_categorical(y_test, 2)
    X_train = (X_train.astype(np.float32) - 127.5)/128.
    X_train = X_train[:,:,:,np.newaxis]

    X_test = np.concatenate([X_test[y_test==0],X_test[y_test==8]],axis=0)
    y_test = np.concatenate([y_test[y_test==0],y_test[y_test==8]-7],axis=0)
    y_test = keras.utils.to_categorical(y_test, 2)
    X_test = (X_test.astype(np.float32) - 127.5)/128.
    X_test = X_test[:,:,:,np.newaxis]
    history = model.fit(X_train,y_train,batch_size=BATCH_SIZE, epochs=200, verbose=1,validation_data=(X_test,y_test))
    plt.plot(history.history['val_acc'])
    plt.title('validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('validation-accuracy.png')
    plt.clf()
    acc_list = history.history['val_acc']
    print('==================================')
    print('The highest validation accuracy is at epoch %d, which is %f'%(acc_list.index(max(acc_list)),max(acc_list)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    num_classes =2
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == 'test':
        test(BATCH_SIZE=args.batch_size)
    elif args.mode == 'train_classifier':
        train_classifier(BATCH_SIZE=args.batch_size)
