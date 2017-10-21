require 'torch'
require 'nn'


---- SoftMax loss
Descriptor:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)