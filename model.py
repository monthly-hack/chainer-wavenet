import chainer.functions as F
import chainer.links as L

class WaveNet(Chain):
    
    ''' WaveNet model.
    
    Args:
        dilations (list of int): Dilations of delated conv.
        residual_channels (int): Dimension of input of x
        dilation_channels (int): Dimension of input of gated activation unit.
        skip_channels (int): Number of channels after skip-connections.
        quantization_channels (int): 
    '''
    
    def __init__(self, dilations,
                 residual_channels=16,
                 dilation_channels=32,
                 skip_channels=128,
                 quantization_channels=256):
        
        super(WaveNet, self).__init__ (
            # a "one-hot" causal conv
            causal_embedID=L.EmbedID(quantization_channels, 2 * residual_channels),
            
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
            
            
    def __call__(self, x):
        ''' Computes the unnormalized log probability.
        It uses L.EmbedID in first causal conv because it is efficient for one-hot input.
        
        Args:
            x (Variable): Variable holding 3 dimensional int32 array whose element 
            indicates quantized amplitude. 
            The shape must be (B, 1, wavelength).
        Returns:
            Variable: A variable holding 4 dimensional float32 array whose element 
            indicates unnormalized log probability.
            The shape is (B, quantization_channels, 1, wavelength - diff_length).        
        ''' 
        
        # a "one-hot" causal conv
        x = self.causal_embedID(x)
        x = x[..., :-1, :self.residual_channels] + x[..., 1:, self.residual_channels:]

        x = F.transpose(x, (0, 3, 1, 2))    # shape=(B, residual_channels, 1, wavelength-1)
        
        # dilated stack and skip connections
        skip = []
        for i in range(len(dilations)):
            out = F.tanh(self['conv_filter{}'.format(i)](x)) * F.sigmoid(self['conv_gate{}'.format(i)](x))
            skip.append(out)
            len_out = out.data.shape[3]
            x = self['conv_res{}'.format(i)](out) + x[:, :, :, -len_out:]
        
        skip = [out[:, :, :, -len_out:] for out in skip]
        y = F.concat(skip)
        
        # last 3 layers
        y = F.relu(self.conv1x1_0(y))
        y = F.relu(self.conv1x1_1(y))
        y = self.conv1x1_2(y)
        
        return y 