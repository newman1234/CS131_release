import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kCenterX = int(Hk / 2)
    kCenterY = int(Wk / 2)

    for ii in range(Hi):
        for ij in range(Wi):
            for ki in range(Hk):
                m = Hk - 1 - ki # flip kernel on row dimension
                for kj in range(Wk):
                    n = Wk - 1 - kj # flip kernel on col dimension

                    i = ii + (m - kCenterY) # calculate row index, kernel cneter == (0,0) 
                    j = ij + (n - kCenterY) # calculate col index, kernel cneter == (0,0)
                    if i >= 0 and i < Hi and j >= 0 and j < Wi:
                        out[ii, ij] += image[i, j] * kernel[m, n]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    left = np.zeros((2*pad_height + H, pad_width))
    up = np.zeros((pad_height, W))
    down = up
    middle = np.concatenate((up, image, down), axis=0)
    right = left
    out = np.concatenate((left, middle, right), axis=1)
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    Wpad, Hpad = Wk, Hk
    if Hk % 2 == 0:
        Hpad = int(Hk / 2)
    else:
        Hpad = int((Hk - 1) / 2)
    if Wk % 2 == 0:
        Wpad = int(Wk / 2)
    else:
        Wpad = int((Wk - 1) / 2)
    imagePad = zero_pad(image, Hpad, Wpad)
    kernelFlip = np.flip(np.flip(kernel, axis=1), axis=0)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(imagePad[i:i+Hk, j:j+Wk] * kernelFlip)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    def im2col_sliding_strided(A, BSZ, stepsize=1):
        # Parameters
        m,n = A.shape
        s0, s1 = A.strides    
        nrows = m-BSZ[0]+1
        ncols = n-BSZ[1]+1
        shp = BSZ[0],BSZ[1],nrows,ncols
        strd = s0,s1,s0,s1

        out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
        return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

    Wpad, Hpad = Wk, Hk
    if Hk % 2 == 0:
        Hpad = int(Hk / 2)
    else:
        Hpad = int((Hk - 1) / 2)
    if Wk % 2 == 0:
        Wpad = int(Wk / 2)
    else:
        Wpad = int((Wk - 1) / 2)

    imagePad = zero_pad(image, Hpad, Wpad)
    kernelFlip = np.flip(np.flip(kernel, axis=1), axis=0)

    out = (kernelFlip.reshape((1, -1)) @ im2col_sliding_strided(imagePad, kernel.shape)).reshape((Hi, Wi))
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(np.flip(g, axis=0), axis=1))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    Wpad, Hpad = Wk, Hk
    if Hk % 2 == 0:
        Hpad = int(Hk / 2)
    else:
        Hpad = int((Hk - 1) / 2)
    if Wk % 2 == 0:
        Wpad = int(Wk / 2)
    else:
        Wpad = int((Wk - 1) / 2)

    imagePad = zero_pad(f, Hpad, Wpad)
    for i in range(Hi):
        for j in range(Wi):
            gNorm = (g - np.mean(g))/np.std(g)
            imageNorm = (imagePad[i:i+Hk, j:j+Wk] - np.mean(imagePad[i:i+Hk, j:j+Wk])) / np.std(imagePad[i:i+Hk, j:j+Wk])
            out[i, j] = np.sum(imageNorm * gNorm)
    ### END YOUR CODE

    return out
