import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import scipy.io.wavfile as wavfile


def gated(x):
    return F.tanh(x[:, :1]) * F.sigmoid(x[:, 1:])


class WaveNet(Chain):
    def __init__(self):
        super(WaveNet3, self).__init__(
            dc00=L.Convolution2D(1, 1, (2, 1), pad=(1, 0), use_cudnn=False),
            dc01=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc02=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc03=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc04=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc05=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc06=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc07=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc08=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc09=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc10=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc11=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc12=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc13=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc14=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc15=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc16=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc17=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc18=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc19=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc20=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc21=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc22=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc23=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc24=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc25=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc26=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc27=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc28=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),
            dc29=L.Convolution2D(None, 2, (2, 1), pad=(1, 0), use_cudnn=False),

            c101=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c102=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c103=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c104=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c105=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c106=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c107=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c108=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c109=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c110=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c111=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c112=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c113=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c114=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c115=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c116=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c117=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c118=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c119=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c120=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c121=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c122=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c123=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c124=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c125=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c126=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c127=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),
            c128=L.Convolution2D(None, 1, (1, 1), use_cudnn=False),

            conv0=L.Convolution2D(None, 128, (1, 1), use_cudnn=False),
            conv1=L.Convolution2D(None, 128, (1, 1), use_cudnn=False),
            conv2=L.Convolution2D(None, 254, (1, 1), use_cudnn=False),
        )

    def __call__(self, x):
        x00 = F.reshape(self.dc00(x)[:, :, :-1], (1, 1, -1, 2))

        a01 = gated(self.dc01(x00)[:, :, :-1])
        x01 = F.reshape(self.c101(a01) + x00, (1, 1, -1, 4))

        a02 = gated(self.dc02(x01)[:, :, :-1])
        x02 = F.reshape(self.c102(a02) + x01, (1, 1, -1, 8))

        a03 = gated(self.dc03(x02)[:, :, :-1])
        x03 = F.reshape(self.c103(a03) + x02, (1, 1, -1, 16))

        a04 = gated(self.dc04(x03)[:, :, :-1])
        x04 = F.reshape(self.c104(a04) + x03, (1, 1, -1, 32))

        a05 = gated(self.dc05(x04)[:, :, :-1])
        x05 = F.reshape(self.c105(a05) + x04, (1, 1, -1, 64))

        a06 = gated(self.dc06(x05)[:, :, :-1])
        x06 = F.reshape(self.c106(a06) + x05, (1, 1, -1, 128))

        a07 = gated(self.dc07(x06)[:, :, :-1])
        x07 = F.reshape(self.c107(a07) + x06, (1, 1, -1, 256))

        a08 = gated(self.dc08(x07)[:, :, :-1])
        x08 = F.reshape(self.c108(a08) + x07, (1, 1, -1, 512))

        a09 = gated(self.dc09(x08)[:, :, :-1])
        x09 = F.reshape(self.c109(a09) + x08, (1, 1, -1, 1))

        a10 = gated(self.dc10(x09)[:, :, :-1])
        x10 = F.reshape(self.c110(a10) + x09, (1, 1, -1, 2))

        a11 = gated(self.dc11(x10)[:, :, :-1])
        x11 = F.reshape(self.c111(a11) + x10, (1, 1, -1, 4))

        a12 = gated(self.dc12(x11)[:, :, :-1])
        x12 = F.reshape(self.c112(a12) + x11, (1, 1, -1, 8))

        a13 = gated(self.dc13(x12)[:, :, :-1])
        x13 = F.reshape(self.c113(a13) + x12, (1, 1, -1, 16))

        a14 = gated(self.dc14(x13)[:, :, :-1])
        x14 = F.reshape(self.c114(a14) + x13, (1, 1, -1, 32))

        a15 = gated(self.dc15(x14)[:, :, :-1])
        x15 = F.reshape(self.c115(a15) + x14, (1, 1, -1, 64))

        a16 = gated(self.dc16(x15)[:, :, :-1])
        x16 = F.reshape(self.c116(a16) + x15, (1, 1, -1, 128))

        a17 = gated(self.dc17(x16)[:, :, :-1])
        x17 = F.reshape(self.c117(a17) + x16, (1, 1, -1, 256))

        a18 = gated(self.dc18(x17)[:, :, :-1])
        x18 = F.reshape(self.c118(a18) + x17, (1, 1, -1, 512))

        a19 = gated(self.dc19(x18)[:, :, :-1])
        x19 = F.reshape(self.c119(a19) + x18, (1, 1, -1, 1))

        a20 = gated(self.dc20(x19)[:, :, :-1])
        x20 = F.reshape(self.c120(a20) + x19, (1, 1, -1, 2))

        a21 = gated(self.dc21(x20)[:, :, :-1])
        x21 = F.reshape(self.c121(a21) + x20, (1, 1, -1, 4))

        a22 = gated(self.dc22(x21)[:, :, :-1])
        x22 = F.reshape(self.c122(a22) + x21, (1, 1, -1, 8))

        a23 = gated(self.dc23(x22)[:, :, :-1])
        x23 = F.reshape(self.c123(a23) + x22, (1, 1, -1, 16))

        a24 = gated(self.dc24(x23)[:, :, :-1])
        x24 = F.reshape(self.c124(a24) + x23, (1, 1, -1, 32))

        a25 = gated(self.dc25(x24)[:, :, :-1])
        x25 = F.reshape(self.c125(a25) + x24, (1, 1, -1, 64))

        a26 = gated(self.dc26(x25)[:, :, :-1])
        x26 = F.reshape(self.c126(a26) + x25, (1, 1, -1, 128))

        a27 = gated(self.dc27(x26)[:, :, :-1])
        x27 = F.reshape(self.c127(a27) + x26, (1, 1, -1, 256))

        a28 = gated(self.dc28(x27)[:, :, :-1])
        x28 = F.reshape(self.c128(a28) + x27, (1, 1, -1, 512))

        a29 = gated(self.dc29(x28)[:, :, :-1])

        a_list = [a01, a02, a03, a04, a05, a06, a07, a08, a09, a10,
                  a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                  a21, a22, a23, a24, a25, a26, a27, a28, a29]
        sc = F.concat([F.reshape(a, (1, 1, -1, 1)) for a in a_list])
        y = self.conv2(F.relu(self.conv1(F.relu(self.conv0(sc)))))
        y = F.swapaxes(y, 1, 3)[0, 0]
        return y


gen_len = 1024 * 30   # Multiple of 1024
scale = 4096.


with aifc.open('mydataset.aiff') as f:
    fs = f.getframerate()
    arr = []
    while True:
        b = f.readframes(1)
        if b == b'':
            break
        arr.append(int.from_bytes(b, 'little', signed=True))


arr = np.array(arr)
arr /= scale

u, indices = np.unique(arr, return_inverse=True)

length = (len(arr) - 1)//1024*1024

x = arr[-length - 1: -1]
x = x.reshape(1, 1, -1, 1).astype(np.float32)
t = indices[-length:].astype(np.int32)
x = cuda.to_gpu(x)
t = cuda.to_gpu(t)
bias = cuda.cupy.array([0, 1], dtype=np.float32)

SCE = lambda y, t: F.softmax_cross_entropy(y, t, use_cudnn=False)
model = L.Classifier(WaveNet3(), SCE)
model.to_gpu()

for i in model.predictor.links():
    if i.name == 'dc00':
        continue
    elif 'dc' in i.name:
        i.b.data += bias

optimizer = optimizers.Adam()
optimizer.setup(model)

#train
print('loss     acc.')
for i in range(50000):
    optimizer.update(model, x, t)
    if i % 100 == 0:
        print('%05.6f %05.6f' % (model.loss.data, model.accuracy.data))

print('saving model and state')
serializers.save_hdf5('model.h5', model)
serializers.save_hdf5('state.h5', optimizer)

# generate
print('generating')

x_sample = Variable(cuda.cupy.zeros((1, 1, gen_len, 1), dtype=np.float32),
                    volatile=True)

for loc in range(x_sample.data.shape[2])[1024:]:
    prob = F.softmax(model.predictor(x_sample[:, :, loc-1024:loc])[-1:],
                     use_cudnn=False).data
    prob = cuda.to_cpu(prob)[0]
    x_sample.data[0, 0, loc] = np.random.choice(u, p=prob)
    if loc % 200 == 0:
        print(loc)

wavfile.write('result.wav', 8000,
              (cuda.to_cpu(x_sample.data)*scale).astype(np.int16).reshape((-1,
                                                                           )))

print('done')
