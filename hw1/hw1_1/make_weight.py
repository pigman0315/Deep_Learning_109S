import numpy as np
l1 = np.random.randn(1024,2048)
l2 = np.random.randn(2048,512)
l3 = np.random.randn(512,128)
print(l1)
op = np.random.randn(128,6)
np.savez('weight_3.npz',layer1=l1,layer2=l2,layer3=l3,output = op)
data = np.load('weight_3.npz')