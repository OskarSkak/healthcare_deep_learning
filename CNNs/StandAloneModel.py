import sys
from matplotlib import pyplot
from numpy.lib.npyio import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import os
import cv2
import glob 
import numpy as np
from PIL import Image

# load train and test dataset
def load_dataset():
	# load dataset
    sys.path.append('./cs231n')
    from data_utils import load_CIFAR10
    cifar10_dir = './cs231n/datasets/cifar-10-batches-py'
    
    trainX, trainY, testX, testY = load_CIFAR10(cifar10_dir) 
	#(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 

def define_model_one_VGG_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_two_VGG_blocks():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def define_model_three_VGG_blocks():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# define cnn model
def define_model_with_dropout_reg_three_VGG_blocks(drop_out_rate_first_block, drop_out_rate_second_block, drop_out_rate_third_block):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_first_block))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_second_block))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_third_block))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# plot diagnostic learning curves
def summarize_diagnostics(filename, history):
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    # filename = sys.argv[0].split('/')[-1]
    pyplot.tight_layout()
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

def save_model(model, name):
    #Since we appended cs231
    sys.path.remove('../cs231n')
    model.save('./models/' + name)

def save_summed_results(details_about_model, acc):
    f = open("./results/summed_res.txt", "a")
    f.write(details_about_model, ", Accuracy: ", acc)
    f.close()



def convert_images_to_arr(images):
    res = []

    for img in images:
        img = Image.open(img)
        rezised_img = img.resize((32, 32))
        data = np.asarray(rezised_img)
        res.append(data)

    return res

def load_images_from_folder(folder):
    res = []
    path = folder + "/*.jpg"
    for img in glob.glob(path):
        res.append(img)

    return convert_images_to_arr(res)

def load_medical_data():
    test_acne_x = load_images_from_folder('./skin_disease_data/test/Acne_and_Rosacea_Photos')
    test_acne_y = np.empty(len(test_acne_x))
    test_acne_y.fill(0)

    test_actinic_x = load_images_from_folder('./skin_disease_data/test/Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions')
    test_actinic_y = np.empty(len(test_actinic_x))
    test_actinic_y.fill(1)

    test_atopic_x = load_images_from_folder('./skin_disease_data/test/Atopic Dermatitis Photos')
    test_atopic_y = np.empty(len(test_atopic_x))
    test_atopic_y.fill(1)

    test_bullous_x = load_images_from_folder('./skin_disease_data/test/Bullous Disease Photos')
    test_bullous_y = np.empty(len(test_bullous_x))
    test_bullous_y.fill(1)

    test_cellulitis_x = load_images_from_folder('./skin_disease_data/test/Cellulitis Impetigo and other Bacterial Infections')
    test_cellulitis_y = np.empty(len(test_cellulitis_x))
    test_cellulitis_y.fill(1)

    test_eczema_x = load_images_from_folder('./skin_disease_data/test/Eczema Photos')
    test_eczema_y = np.empty(len(test_eczema_x))
    test_eczema_y.fill(1)

    test_exanthems_x = load_images_from_folder('./skin_disease_data/test/Exanthems and Drug Eruptions')
    test_exanthems_y = np.empty(len(test_exanthems_x))
    test_exanthems_y.fill(1)

    test_hair_loss_alopecia_x = load_images_from_folder('./skin_disease_data/test/Hair Loss Photos Alopecia and other Hair Diseases')
    test_hair_loss_alopecia_y = np.empty(len(test_hair_loss_alopecia_x))
    test_hair_loss_alopecia_y.fill(1)

    test_herpes_x = load_images_from_folder('./skin_disease_data/test/Herpes HPV and other STDs Photos')
    test_herpes_y = np.empty(len(test_herpes_x))
    test_herpes_y.fill(1)

    test_light_diseases_x = load_images_from_folder('./skin_disease_data/test/Light Diseases and Disorders of Pigmentation')
    test_light_diseases_y = np.empty(len(test_light_diseases_x))
    test_light_diseases_y.fill(1)

    test_lupus_x = load_images_from_folder('./skin_disease_data/test/Lupus and other Connective Tissue diseases')
    test_lupus_y = np.empty(len(test_lupus_x))
    test_lupus_y.fill(1)

    test_melanoma_x = load_images_from_folder('./skin_disease_data/test/Melanoma Skin Cancer Nevi and Moles')
    test_melanoma_y = np.empty(len(test_melanoma_x))
    test_melanoma_y.fill(1)
    
    test_nail_fungus_x = load_images_from_folder('./skin_disease_data/test/Nail Fungus and other Nail Disease')
    test_nail_fungus_y = np.empty(len(test_nail_fungus_x))
    test_nail_fungus_y.fill(1)

    test_poison_ivy_x = load_images_from_folder('./skin_disease_data/test/Poison Ivy Photos and other Contact Dermatitis')
    test_poison_ivy_y = np.empty(len(test_poison_ivy_x))
    test_poison_ivy_y.fill(1)

    test_psoriasis_x = load_images_from_folder('./skin_disease_data/test/Psoriasis pictures Lichen Planus and related diseases')
    test_psoriasis_y = np.empty(len(test_psoriasis_x))
    test_psoriasis_y.fill(1)

    test_scabies_lyme_x = load_images_from_folder('./skin_disease_data/test/Scabies Lyme Disease and other Infestations and Bites')
    test_scabies_lyme_y = np.empty(len(test_scabies_lyme_x))
    test_scabies_lyme_y.fill(1)

    test_seborrheic_keratoses_x = load_images_from_folder('./skin_disease_data/test/Seborrheic Keratoses and other Benign Tumors')
    test_seborrheic_keratoses_y = np.empty(len(test_seborrheic_keratoses_x))
    test_seborrheic_keratoses_y.fill(1)

    test_systemic_disease_x = load_images_from_folder('./skin_disease_data/test/Systemic Disease')
    test_systemic_disease_y = np.empty(len(test_systemic_disease_x))
    test_systemic_disease_y.fill(1)

    test_tinea_ringworm_x = load_images_from_folder('./skin_disease_data/test/Tinea Ringworm Candidiasis and other Fungal Infections')
    test_tinea_ringworm_y = np.empty(len(test_tinea_ringworm_x))
    test_tinea_ringworm_y.fill(1)

    test_urticaria_hives_x = load_images_from_folder('./skin_disease_data/test/Urticaria Hives')
    test_urticaria_hives_y = np.empty(len(test_urticaria_hives_x))
    test_urticaria_hives_y.fill(1)

    test_vascular_tumors_x = load_images_from_folder('./skin_disease_data/test/Vascular Tumors')
    test_vascular_tumors_y = np.empty(len(test_vascular_tumors_x))
    test_vascular_tumors_y.fill(1)

    test_vasculitis_x = load_images_from_folder('./skin_disease_data/test/Vasculitis Photos')
    test_vasculitis_y = np.empty(len(test_vasculitis_x))
    test_vasculitis_y.fill(1)

    test_warts_molluscum_x = load_images_from_folder('./skin_disease_data/test/Warts Molluscum and other Viral Infections')
    test_warts_molluscum_y = np.empty(len(test_warts_molluscum_x))
    test_warts_molluscum_y.fill(1)

    
    # acneY = #Np.array af samme længde, og med værdi 1 for hver
    return convert_images_to_arr(acne)
 
def run_test_harness():

    load_medical_data()
    # sys.path.append('../skin_disease_data')
    # print(os.system("dir"))
    # acne = load_medical_data()
    # print(type(acne))
    # print(acne.shape)

    a = ""

    # print(" > Loading dataset...")
    # trainX, trainY, testX, testY = load_dataset()
    # a = ""
    # print(" > Preprocessing dataset...")
    # trainX, testX = prep_pixels(trainX, testX)

    # print("Shapes:")
    # print("Train xy: ", trainX.shape, ", ", trainY.shape)
    # print("Test xy: ", testX.shape, ", ", testY.shape)
    
    # print(" > Dataset loaded...")
    # print(" > Defining model...")
    # #model = define_model_one_VGG_block()
    # model = define_model_two_VGG_blocks()
    
    # print(" > Fitting model...")
    # history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
    # print(" > Model finished fitting...")

    # print(" > Evaluating model...")
    # _, acc = model.evaluate(testX, testY, verbose=0)
    # print(" > Accuracy: ", '> %.3f' % (acc * 100.0))
    
    # print(" > Summarizing diagnostics and saving model...")
    # summarize_diagnostics('two_VGG_epoch100_batch64', history)
    # save_model(model, 'two_VGG_epoch100_batch64')
    # save_summed_results('two_VGG_epoch100_batch64', acc)
 
# entry point, run the test harness
run_test_harness()