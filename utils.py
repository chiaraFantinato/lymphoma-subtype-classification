from tensorflow.keras.models import save_model
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib as plt

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def rgb2gray(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return gray

def scaling(img):
    return ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)

def var_of_laplacian(image): # high better --> threshold = 100 but I use it in reverse
    return round(cv2.Laplacian(image,cv2.CV_64F).var(),2)

def brisque_score(image):
    return round(BRISQUE(url=False).score(image),2)

class ModelSaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        save_model(self.model, self.file_name)
        print("Epoch {} - Model saved in {}".format(epoch, self.file_name))
        
def heatmap_on_image(input_img,output_layer,threshold=0.7):
    output = cv2.resize(output_layer[0], dsize=(input_img.shape[0], input_img.shape[1]), interpolation=cv2.INTER_CUBIC)
    plt.figure(figsize=(14,12))
    plt.subplot(161); plt.imshow(input_img); plt.title('original')
    plt.subplot(162); plt.imshow(output_layer[0],cmap='jet'); plt.title('avg pool of last conv2d')
    plt.subplot(163); plt.imshow(output,cmap='jet'); plt.title('with image size')
    plt.subplot(164); plt.imshow(output>=threshold,cmap='gray'); plt.title('with image size')
    axes = plt.subplot(165); 
    axes.imshow(Image.fromarray(input_img.astype(np.uint8)).convert('L'), cmap='gray'); plt.title('overlaid')
    axes.imshow(output, alpha = 0.3, cmap='jet', interpolation='bilinear')
    img_zeros = np.zeros((input_img.shape))
    img_zeros[output>=threshold,:]=input_img[output>=threshold,:]
    plt.subplot(166); plt.imshow(img_zeros.astype(np.uint8))
    
def extract_maskedimage(input_img,heatmap,threshold=0.7):
    scaled = scaling01(heatmap); #print(scaled.shape)
    output = cv2.resize(scaled, dsize=(input_img.shape[0], input_img.shape[1]), interpolation=cv2.INTER_CUBIC)
    img_zeros = np.zeros(input_img.shape)
    img_zeros[output>=threshold,:]=input_img[output>=threshold,:]
    masked_heatmap = img_zeros.astype(int)
    return masked_heatmap

def create_masked_ds(ds_original,heatmaps):
    ds_heatmap = []
    for k in range(ds_original.shape[0]):
        ds_heatmap.append(extract_maskedimage(ds_original[k],heatmaps[k]))
    return np.array(ds_heatmap)

def scaling01(array):
    return (array-array.min())/(array.max()-array.min())
