import argparse
import numpy as np
from unittest import mock

import chainer
import chainer.functions as F
import chainer.links as L

import onnx_chainer


class MLP(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(3, 4)
            self.fc2 = L.Linear(4, 5)

        self.fc1.W.array[:] = \
            np.arange(0, 12, dtype=np.float32).reshape((4, 3))
        self.fc1.b.array[:] = np.arange(12, 16, dtype=np.float32)

        self.fc2.W.array[:] = \
            np.arange(16, 36, dtype=np.float32).reshape((5, 4))
        self.fc2.b.array[:] = np.arange(36, 41, dtype=np.float32)

    def __call__(self, x):
        x.node._onnx_name = 'input'
        h = F.relu(self.fc1(x))
        h.node._onnx_name = 'fc1'
        h = F.relu(self.fc2(h))
        h.node._onnx_name = 'fc2'
        return h


class IDGenerator(object):

    def __init__(self):
        # keep original
        self._id = id

    def __call__(self, obj):
        return getattr(obj, '_onnx_name', self._id(obj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    args = parser.parse_args()

    model = MLP()
    x = np.empty((1, 3), dtype=np.float32)
    with chainer.using_config('train', False), \
            mock.patch('builtins.id', IDGenerator()):
        onnx_chainer.export(model, x, filename=args.out)


if __name__ == '__main__':
    main()
