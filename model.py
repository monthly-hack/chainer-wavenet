import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import reporter


class WaveNet(Chain):

    ''' Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(10)] * 3
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        model = WaveNet(dilations, residual_channels, dilation_channels, skip_channels,
                        quantization_channels)
    '''

    def __init__(self, dilations,
                 residual_channels=16,
                 dilation_channels=32,
                 skip_channels=128,
                 quantization_channels=256):
        '''
        Args:
            dilations (list of int): 
                A list with the dilation factor for each layer.
            residual_channels (int): 
                How many filters to learn for the residual.
            dilation_channels (int): 
                How many filters to learn for the dilated convolution.
            skip_channels (int): 
                How many filters to learn that contribute to the quantized softmax output.
            quantization_channels (int): 
                How many amplitude values to use for audio quantization and the corresponding 
                one-hot encoding.
                Default: 256 (8-bit quantization).
        '''

        super(WaveNet, self).__init__(
            # a "one-hot" causal conv
            causal_embedID=L.EmbedID(
                quantization_channels, 2 * residual_channels),

            # last 3 layers (include convolution on skip-connections)
            conv1x1_0=L.Convolution2D(None, skip_channels, 1),
            conv1x1_1=L.Convolution2D(None, skip_channels, 1),
            conv1x1_2=L.Convolution2D(None, quantization_channels, 1),
        )
        # dilated stack
        for i, dilation in enumerate(dilations):
            self.add_link('conv_filter{}'.format(i),
                          L.DilatedConvolution2D(None, dilation_channels, (1, 2), dilate=dilation))
            self.add_link('conv_gate{}'.format(i),
                          L.DilatedConvolution2D(None, dilation_channels, (1, 2), dilate=dilation, bias=1))
            self.add_link('conv_res{}'.format(i),
                          L.Convolution2D(None, residual_channels, 1, nobias=True))

        self.residual_channels = residual_channels
        self.dilations = dilations

    def __call__(self, x):
        ''' Computes the unnormalized log probability.
        It uses L.EmbedID in first causal conv because it is efficient for one-hot input.

        Args:
            x (Variable): Variable holding 3 dimensional int32 array whose element indicates
            quantized amplitude. 
            The shape must be (B, 1, wavelength).
        Returns:
            Variable: A variable holding 4 dimensional float32 array whose element indicates
            unnormalized log probability.
            The shape is (B, quantization_channels, 1, wavelength - ar_order + 1).        
        '''

        # a "one-hot" causal conv
        x = self.causal_embedID(x)
        x = x[..., :-1, :self.residual_channels] + \
            x[..., 1:, self.residual_channels:]

        # shape (B, residual_channels, 1, wavelength-1)
        x = F.transpose(x, (0, 3, 1, 2))

        # dilated stack and skip connections
        skip = []
        for i in range(len(self.dilations)):
            out = F.tanh(self['conv_filter{}'.format(i)](x)) * \
                F.sigmoid(self['conv_gate{}'.format(i)](x))
            skip.append(out)
            len_out = out.data.shape[3]
            x = self['conv_res{}'.format(i)](out) + x[..., -len_out:]

        skip = [out[:, :, :, -len_out:] for out in skip]
        y = F.concat(skip)

        # last 3 layers
        y = F.relu(self.conv1x1_0(y))
        y = F.relu(self.conv1x1_1(y))
        y = self.conv1x1_2(y)

        return y


class ARClassifier(Chain):

    compute_accuracy = True

    def __init__(self, predictor, ar_order,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy):
        super(ARClassifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        self.ar_order = ar_order

    def __call__(self, arg):
        x = arg[..., :-1]
        t = arg[..., self.ar_order:]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
