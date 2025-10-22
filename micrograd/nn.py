import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
# Neuron Initialization 
# random weights and bias  initialized for each neuron inputs
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

#Num of Neuron Initialization done
#Layer(2,16,nonlin=True) <> 16 neurons with 2 inputs x1,x2
#Layer(16,16,nonlin=True) <> 16 neurons with 16 inputs xx1,xx2,xx3....xx16
#Layer(16,1,nonlin=False) <> 1 neuron with 16 inputs
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

# MLP(2, [16, 16, 1])
#nin = 2 ~ 2 inputs x1 and x2
#nouts = [16, 16, 1] ~ 3 layers 
#sz = [2, 16, 16, 1]
#Layer initialization
#Layer 1 2 inputs <> 16 outputs Layer(2,16,nonlin=True)
#Layer 2 16 inputs <> 16 outpus Layer(16,16,nonlin=True)
#Layer 3 16 outputs <> 1 output Layer(16,1,nonlin=False)

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
