import numpy as np
from derivata.nn import Linear
import derivata

x = derivata.Variable(np.random.randn(5,5,1))
net = Linear(5,10)

z = net(x)

z.backward(root=True)

#print(w2.total_derivative)
#print(w.total_derivative)
print(W.total_derivative)
print(x.total_derivative)
#print(y.total_derivative)
#print(x.total_derivative)

