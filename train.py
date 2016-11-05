# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import cuda, Variable
from chainer import datasets, iterators, optimizers, serializers, training
import chainer.functions as F
from chainer.training import extensions
from chainer.dataset import iterator

import scipy.io.wavfile as wavfile
import os, librosa, fnmatch

from model import *


directory = 'dataset/dateset/'

sample_rate = 8000

output_file_dir = 'results/'
output_len = 100000
gpu = 0
resume = False
epoch = 100
train_length = 10000

residual_channels = 16
dilation_channels = 32
skip_channels = 16
dilations = [2**i for i in range(10)] * 3

quantization_channels = 255


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = quantization_channels - 1
    # Perform mu-law companding transformation (ITU-T, 1988).
    magnitude = np.log(1 + mu * np.abs(audio)) / np.log(1. + mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    casted = output.astype(np.float32)
    signal = 2. * (casted / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude


def chop_dataset(data, train_length, stride, ar_order):
    k = train_length + ar_order
    dataset = np.stack([data[stride * i : stride * i + k] 
                         for i in range((len(data) - k) // stride + 1)])
    return dataset[:, np.newaxis, :, 0]


def generate_and_write_one_sample(ar_order, x, loc):
    y = model.predictor(x[..., loc - ar_order : loc])
    
    prob = F.softmax(y).data.flatten()
    prob = cuda.to_cpu(prob)
    x.data[..., loc] = np.random.choice(range(quantization_channels), p=prob)


def save_x(x, ar_order, quanttization_channels, filename, fs):
    output = mu_law_decode(cuda.to_cpu(x.data[0, 0, ar_order:]), quantization_channels)
    output = np.round(output * 2 ** 15).astype(np.int16).reshape((-1,))
    wavfile.write(filename, fs, output)



ar_order = sum(dilations) + 2

wave_arrays = []
for audio, _ in load_generic_audio(directory, sample_rate):
    x = mu_law_encode(audio, quantization_channels)
    x = chop_dataset(x, train_length, train_length, ar_order)
    wave_arrays.append(x)

dataset = np.concatenate(wave_arrays).astype(np.int32)

if gpu >= 0:
    cuda.get_device(gpu).use() 



model = ARClassifier(WaveNet(dilations, 
                             residual_channels,
                             dilation_channels,
                             skip_channels,
                             quantization_channels),
                    ar_order)
model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)


train, test = chainer.datasets.split_dataset_random(dataset, len(dataset) // 10 * 9)

train_iter = chainer.iterators.SerialIterator(train, 6)
test_iter = chainer.iterators.SerialIterator(test, 8, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu)


trainer = training.Trainer(updater, (epoch, 'epoch'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

trainer.extend(extensions.dump_graph('main/loss'))

trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

trainer.extend(extensions.ProgressBar())

trainer.run()
