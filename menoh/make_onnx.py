import argparse
import numpy as np

import onnx
import onnx.numpy_helper


def make_tensor_value_info(name, shape):
    return onnx.helper.make_tensor_value_info(
        name, onnx.TensorProto.FLOAT, shape)


def from_array(name, array):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    args = parser.parse_args()

    inputs = (
        make_tensor_value_info('input', (3,)),
        make_tensor_value_info('fc1_W', (4, 3)),
        make_tensor_value_info('fc1_b', (4,)),
        make_tensor_value_info('fc2_W', (5, 4)),
        make_tensor_value_info('fc2_b', (4,)),
    )

    outputs = (
        make_tensor_value_info('fc1', (4,)),
        make_tensor_value_info('fc2', (5,)),
    )

    nodes = (
        onnx.helper.make_node(
            'Gemm', ('input', 'fc1_W', 'fc1_b'), ('_fc1',),
            alpha=1., beta=1., transA=0, transB=1),
        onnx.helper.make_node('Relu', ('_fc1',), ('fc1',)),
        onnx.helper.make_node(
            'Gemm', ('fc1', 'fc2_W', 'fc2_b'), ('_fc2',),
            alpha=1., beta=1., transA=0, transB=1),
        onnx.helper.make_node('Relu', ('_fc2',), ('fc2',)),
    )

    initializers = (
        from_array('fc1_W', np.arange(-6, 6).reshape((4, 3))),
        from_array('fc1_b',  np.arange(-2, 2)),
        from_array('fc2_W', np.arange(-10, 10).reshape((5, 4))),
        from_array('fc2_b',  np.arange(-2, 3)),
    )

    graph = onnx.helper.make_graph(
        nodes, 'model', inputs, outputs, initializer=initializers)
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)


if __name__ == '__main__':
    main()
