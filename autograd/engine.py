import math


class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self._backward = lambda: None
        self.grad = 0.0
        self._prev = set(_children)
    
    def __repr__(self):
        return str(self.data)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out 
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value((self.data ** other), (self,))

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        tanh_data = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(tanh_data, (self,))

        def _backward():
            self.grad += (1 - tanh_data**2)*out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        sorted_nodes = []
        visited = set()
        def topo_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topo_sort(child)
                sorted_nodes.append(node)
        topo_sort(self)
        
        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backward()