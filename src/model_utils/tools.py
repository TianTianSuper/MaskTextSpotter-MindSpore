from mindspore import nn
from mindspore.ops import operations as P

class SafeConcat(nn.Cell):
    def __init__(self, axis=0):
        super(SafeConcat, self).__init__()
        self.concat = P.Concat(axis=axis)
    
    def construct(self, tensors):
        assert isinstance(tensors,(list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return self.concat(tensors)