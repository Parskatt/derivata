import numpy as np
from derivata import Variable

x = Variable(np.random.randn(5,5))
y = Variable(np.random.randn(5,5))

z = x+y
w = z-y
w2 = w*w
x.name = "x"
y.name = "y"
z.name = "z"
w.name = "w"
w2.name = "w2"

w2.backward(root=True)

#print(w2.total_derivative)
#print(w.total_derivative)
print(z.total_derivative)
#print(y.total_derivative)
#print(x.total_derivative)

