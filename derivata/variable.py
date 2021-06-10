import numpy as np
from derivata import add, subtract, mult, matmul

def assure_variable(f):
    def f_var(self,other):
        other = other if isinstance(other, Variable) else Variable(other)
        return f(self,other)
    return f_var

class Variable:
    def __init__(self,*args,func=None):
        if func:
            self.data = func(*args)
        else:
            self.data = np.array(args[0])
        self.func = func
        self.args = args
        self.grad = 0.
        self.total_derivative = 0.
        self.forward_edges = set()
        self.shape = self.data.shape

    def requires_grad_(self):
        self.requires_grad = True

    def zero_grad(self):
        self.grad = 0.
        self.total_derivative = 0.
        if self.forward_edges != set():
            raise UserWarning('zero_grad with forward edges remaining.')

    def set_grad(self,grad):
        self.grad = grad
        self.total_derivative = self.total_derivative+self.grad

    @assure_variable
    def __add__(self, other):
        return Variable(self,other,func=add) if self is not other else Variable(self,2,func=mult)
    __radd__ = __add__

    @assure_variable
    def __sub__(self, other):
        return Variable(self,other,func=subtract) if self is not other else Variable(self,-self,func=add)
    __rsub__ = __sub__
    def __neg__(self):
        return Variable(self*-1.)

    @assure_variable
    def __mul__(self, other):
        return Variable(self,other,func=mult) if self is not other else Variable(self,1.*self,func=mult)
    __rmul__ = __mul__

    def __matmul__(self,other):
        return Variable(self,other,func=matmul)

    def t(self):
        return Variable(self.data.T)

    def backward(self,root=False):
        if self.func is None:
            return
        if root:
            self.set_grad(Variable(np.ones_like(self.data)))
            dz = self.total_derivative
            self.traverse_graph()
        else:
            if self.forward_edges != set():
                return
            dz = self.total_derivative#self.grad
        
        self.func.vjp(dz,*self.args)
        for arg in self.args:
            arg.forward_edges.remove(id(self))
            arg.backward()
    
    def traverse_graph(self):
        if self.func == None:
            return
        for arg in self.args:
            arg.forward_edges.add(id(self))
            arg.traverse_graph()

    def __str__(self):
        return f"data is {self.data},created by {self.func}"
    __repr__ = __str__