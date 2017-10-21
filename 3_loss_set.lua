require 'torch'
require 'nnx'


---- SoftMax loss
--net:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)