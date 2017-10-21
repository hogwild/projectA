require 'torch'
require 'xlua'
require 'optim'
--require 'cuda'

-- temp data for testing this function
--[[require 'nnx'
net = nn.Sequential()
net:add(nn.Linear(10,5))
net:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()--]]
--trainIm=torch.rand(100,10,10,10)
--trainLabel = torch.Tensor(100):fill(1)



----CUDA
if opt.type == 'cuda' then
   net:cuda()
   criterion:cuda()
end

----training 
print '==> defining training procedure'
--classes
classes = {'1','2','3','4','5','6','7','8','9','10','11','12', '13','14' ,'15', '16', '17', '18', '19', '20', '21','22','23','24', '25', '26','27','28','29','30','31','32','33','34'}
--cofusion matrix
confusion = optim.ConfusionMatrix(classes)
--log results to files
trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

----retrive parameters and gradients:
if net then
  parameters, gradParameters = net:getParameters()
end

---- the size of the train set
local trsize = trainLabel:size(1)
----training function
function train()
  epoch = epoch or 1
  local time = sys.clock()
  shuffle = torch.randperm(trsize)
  --do one epoch
  print('==> doing epoch on training data:')
  print('==>online epoch # '..epoch..'[batchSize = '..opt.batchSize..']')
  for t=1, trainIm:size(1), opt.batchSize do
    xlua.progress(t,trainIm:size(1))
    --create mini batch
    local inputs = {}
    local targets = {}
    for i = t, math.min(t+opt.batchSize-1,trainIm:size(1)) do
      local input = trainIm[shuffle[i]]
      local target = trainLabel[shuffle[i]]
      if opt.type == 'cuda' then input = input:cuda() end
      table.insert(inputs,input)
      table.insert(targets,target)
    end
    ----create closure to evaluate f(X) and df/dX
    local feval = function(x)
      --get new parameters
      if x~=parameters then
        parameters:copy(x)
      end
      ----reset gradients
      gradParameters:zero()
      ----f is the average of all criterions, f_p is the total criterions of all pixels in one image
      local f = 0
      local f_p=0
      ----evaluation funtion for complete mini patch
      for i = 1, #inputs do
        ----estimate f
        local output = net:forward(inputs[i]) ---- the output is a m x n matrix, m is the number of pixels in a image (in our case it is 256*256=65536), n is the number of classes (here is 34). So, we have to compute the err of each pixel. Iteration of pixels:
        local df_do = torch.Tensor():resizeAs(output)
        for p = 1, output:size(1) do
          local err = criterion:forward(output[p], targets[i][p])
          f_p = f_p+err
        ---- estimate df/dw
          df_do[p] = criterion:backward(output[p],targets[i][p])
        ----update confusion
        confusion:add(output[p],targets[i][p]) ----in fact, the confusion is based on each pixel
        end
        f = f_p/output:size(1) ---- the f of the image is considered as the average of f_p (i.e. the f of each pixel)
        net:backward(inputs[i],df_do)
      end
      gradParameters:div(#inputs)
      f=f/#inputs
      return f, gradParameters
    end
    ----optimize on current mini-batch
    if opt.optimization == 'LBFGS' then
      config = config or {learningRate=opt.learningRate,maxIter = opt.maxIter,nCorrection =34}
      optim.lbfgs(feval,parameters,config)
    elseif opt.optimization == 'SGD' then
      config=config or {learningRate = opt.learningRate, weightDecay = opt.weightDecay, momentum = opt.momentum, learningRateDecay = 5e-7}
      optim.sgd(feval, parameters,config)
    else 
      error('unknow optimization method')
    end
  end
  
  ----time taken
  time = sys.clock()-time
  time = time/trainIm:size(1)
  print('==>time to learn 1 sample = '..(time*1000)..'ms')
  ---print confusion matrix
  print(confusion)
  confusion:zero()
  ----update logger/plot
  trainLogger:add{['%mean class accuracy(train set)']=confusion.totalValid*100}
  if opt.plot then
    trainLogger:style{['%mean class accuracy(train set)']='-'}
    trainLogger:plot()
  end
----save/log current net
local filename = paths.concat(opt.save, 'FeatureExtractor.net')
os.execute('mkdir -p'..sys.dirname(filename))
print('==>saving net to '..filename)
torch.save(filename, net)
----next epoch
epoch = epoch+1
end
