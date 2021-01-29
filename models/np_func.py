import numpy as np

def Conv2d(X,W,stride=1,padding=1):
    X_pad = np.zeros([X.shape[0], X.shape[1], (X.shape[2]+2*padding), (X.shape[3]+2*padding)], dtype=type(X))
    print ('X_pad.shape', X_pad.shape)
    print ('X.shape', X.shape)
    X_pad[:,:,padding:X_pad.shape[2]-padding,padding:X_pad.shape[2]-padding] = X
    Y = np.zeros([X.shape[0],W.shape[0],(X.shape[2]+2*padding-W.shape[2]+1)//stride,(X.shape[3]+2*padding-W.shape[3]+1)//stride],dtype=type(X))
    for b in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            for h in range(Y.shape[2]):
                for w in range(Y.shape[3]):
                    Y[b][c][h][w] = np.sum(X_pad[b,:,h*stride:h*stride+W.shape[2],w*stride:w*stride+W.shape[2]]*W[c,:,:,:])
    return Y

def ConvTranspose2d(X, W, stride=2, padding=1):

    bch, cha, h, w = W.shape
    Y = np.zeros((X.shape[0], 
        W.shape[1], 
        (X.shape[2] - 1) * stride + h,
        (X.shape[3] - 1) * stride + w))
    #print ('Y', Y.shape)

    for b in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            for x_cha in range(X.shape[1]):
                for i in range(X.shape[2]):
                    for j in range(X.shape[3]):
                        #print ('W.shape', W.shape)
                        #print ('X.shape', X.shape)
                        # print ('X[:, :, i, j]', X[:, :, i, j] * W)
                        #print ('Y slice', Y[:, :, i: (i + h), j: (j + w)].shape)
                        #print ('i, j', i , j)
                        #print ('Y slice',Y[b, c, i*stride: i*stride + h, j*stride: j*stride + w].shape)
                        #print ('X slice',X[b, x_cha, i, j].shape, X[b, x_cha, i, j])
                        #print ('W slice',W[x_cha, c, :, :].shape)
                        Y[b, c, i*stride: i*stride + h, j*stride: j*stride + w] += X[b, x_cha, i, j] * W[x_cha, c, :, :]
                    # Y[:, :, i: (i + h), j: (j + w)] += X[:, :, i, j] * W
    if padding ==0:
        return Y
    else:
        return Y[:, :, padding:-padding, padding:-padding]

def Linear(X,W,bias=None):
    X = X.reshape(X.shape[0],1,-1)
    Y = np.zeros([X.shape[0],W.shape[0]],dtype=type(X))
    for b in range(X.shape[0]):
        Y[b,:] = W.dot(X[b,0,:])
    if bias==None :
        return Y
    else:
        return Y + bias

def load_weight(file_name,weight_size):
    weight = np.genfromtxt(file_name,delimiter=' ',dtype=None)
    weight = weight.reshape(weight_size)
    return weight

def Act(input_):
    input_[input_ < 0] = 0
    return input_

def Act_Q(input,bit=32):
    n = float(2.0**bit)
    output = input.copy()
    output = np.floor(output * n) / n
    output[output<0] = 0
    output[output>=(1.0-1.0/n)] = (1.0-1.0/n) 
    
    return output

class conv(object):
    def __init__(self, weight=None, stride=1, padding=1):
        # self.input_feature = input_feature
        self.weight = weight
        self.stride = stride
        self.padding = padding
    def fw(self, input_feature=None):
        return Act(Conv2d(input_feature, self.weight, stride=self.stride, padding=self.padding))

class deconv(object):
    def __init__(self, weight=None):
        # self.input_feature = input_feature
        self.weight = weight
    def fw(self, input_feature=None):
        return Act(ConvTranspose2d(input_feature, self.weight, stride=2, padding=1))

class predict_flow(object):
    def __init__(self, weight=None):
        self.weight = weight
    def fw(self, input_feature=None):
        return Conv2d(input_feature, self.weight, stride=1, padding=1)

class upsample(object):
    def __init__(self, weight=None):
        self.weight = weight
    def fw(self, input_feature=None):
        return ConvTranspose2d(input_feature, self.weight, stride=2, padding=1)

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]