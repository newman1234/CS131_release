import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * image ** 2
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = np.dot(image[...,:3], [0.299, 0.587, 0.144])
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    h = image.shape[0]
    w = image.shape[1]
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    if channel == 'R':
        r = np.zeros((h, w))
    elif channel == 'G':
        g = np.zeros((h, w))
    elif channel == 'B':
        b = np.zeros((h, w))
    else:
        print('Input channel is not RGB!')
        
    out = np.stack([r, g, b], axis=2)  
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    if channel == 'L':
        out = lab[..., 0]
    elif channel == 'A':
        out = lab[..., 1]
    elif channel == 'B':
        out = lab[..., 2]
    else:
        print('Input channel is not RGB!')    
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    if channel == 'H':
        out = hsv[..., 0]
    elif channel == 'S':
        out = hsv[..., 1]
    elif channel == 'V':
        out = hsv[..., 2]
    else:
        print('Input channel is not RGB!')
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    h = image1.shape[0]
    w = image1.shape[1]
    r1, g1, b1 = image1[:,:int(w/2),0], image1[:,:int(w/2),1], image1[:,:int(w/2),2]
    r2, g2, b2 = image2[:,int(w/2):,0], image2[:,int(w/2):,1], image2[:,int(w/2):,2]
    if channel1 == 'R':
        r1 = np.zeros((h, int(w/2)))
    elif channel1 == 'G':
        g1 = np.zeros((h, int(w/2)))
    elif channel1 == 'B':
        b1 = np.zeros((h, int(w/2)))
    else:
        print('Input channel1 is not RGB!')
        
    if channel2 == 'R':
        r2 = np.zeros((h, int(w/2)))
    elif channel2 == 'G':
        g2 = np.zeros((h, int(w/2)))
    elif channel2 == 'B':
        b2 = np.zeros((h, int(w/2)))
    else:
        print('Input channel2 is not RGB!')   
        
    out = np.concatenate((np.stack([r1, g1, b1], axis=2), np.stack([r2, g2, b2], axis=2)), axis=1)
    ### END YOUR CODE

    return out
