require 'torch'


print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
cmd:option('-optimization', 'SGD', 'optimization method: SGD|LBFGS')
cmd:option('-learningRate',1e-3,'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)' )
cmd:option('weightDecay', 1e-5, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum(SGD only)')
cmd:option('-maxIter', 2, 'maximum nb of iterations for LBFGS')
cmd:option('-type','cuda', 'type: double| float | cuda')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
  print ('==> switching to CUDA')
  require 'cunn'
  torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'float' then
  print('==> swithching to floats')
  torch.setdefaulttensortype('torch.FloatTensor')
else
  print('==>Warning: not enough memory to run in doubles ')
end

print '==> executing all'

dofile '1_data_preprocess.lua'
--dofile '2_net_build.lua'
dofile '2_load_net.lua'
dofile '3_loss_set.lua'
dofile '4_train_net.lua'
dofile '5_test_net.lua'

print '==> training!'

while true do
   train()
   test()
end--]]
