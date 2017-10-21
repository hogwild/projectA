require 'torch'

trainIm=torch.rand(3, 3,100,100)
testIm=torch.rand(3,3,100,100)
trainLabel = torch.Tensor(3, 100*100):fill(1)
testLabel = torch.Tensor(3,100*100):fill(1)