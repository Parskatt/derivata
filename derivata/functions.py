
class Add:
    @staticmethod
    def __call__(x, y):
        return x.data+y.data
    @staticmethod
    def vjp(dz, x, y):
        x.set_grad(dz)
        y.set_grad(dz)
    def __str__(self):
        return "Add"

class Subtract:
    @staticmethod
    def __call__(x, y):
        return x.data-y.data
    @staticmethod
    def vjp(dz, x, y):
        x.set_grad(dz)
        y.set_grad(-dz)
    def __str__(self):
        return "Subtract"

class Mult:
    @staticmethod
    def __call__(x, y):
        return x.data*y.data
    @staticmethod
    def vjp(dz, x, y):
        x.set_grad(dz*y)
        y.set_grad(dz*x)
    def __str__(self):
        return "Mult"

class ReLU:
    @staticmethod
    def __call__(x):
        return max(x.data,0.)
    @staticmethod
    def vjp(dz, x, y):
        x.set_grad(dz*(x>0))
        y.set_grad(dz*(y>0))
    def __str__(self):
        return "ReLU"
        
class MatMul:
    @staticmethod
    def __call__(W,x):
        return W.data@x.data
    @staticmethod
    def vjp(dz, W, x):
        x.set_grad(dz@W.t())
        W.set_grad(dz@x.t())
    def __str__(self):
        return "MatMul"

add = Add()
subtract = Subtract()
mult = Mult()
relu = ReLU()
matmul = MatMul()