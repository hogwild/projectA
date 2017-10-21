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
classes = {'1','5',	'6',	'7',	'9',	'10',	'12',	'13',	'14',	'15',	'16',	'18',	'19',	'20',	'21',	'22',	'23',	'24',	'25',	'26',	'28',	'29',	'32',	'33',	'34'}
-- [[1(0): background, 2(5): boat, 3(6): bridge, 4(7): building, 5(9): car, 6(10): cow, 7(12):desert, 8(13):door 9(14):fence, 10(15):field, 11(16):grass, 12(18):mountain, 13(19):person, 14(20):plant, 15(21):pole, 16(22):river, 17(23):road, 18(24):rock, 19(25):sand, 20(26):sea, 21(28):sign, 22(29):sky, 23(32): sun, 24(33):tree, 25('34'):window--]]
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
----training function  ----IMPORTATN NOTES: In the train function, "if ...then ...else...end" won't work. Only  if ...then...end will work
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
      local target = trainLabel[shuffle[i]]:view(-1)
      --local input = trainIm[i]
      --local target = trainLabel[i]:view(-1)
      if opt.type == 'cuda' then target = target:cuda() end
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
     -- local f_p=0
      ----evaluation funtion for complete mini patch
      for i = 1, #inputs do
        ----estimate f
        local output = net:forward(inputs[i]) ---- the output is a m x n matrix, m is the number of pixels in a image (in our case it is 256*256=65536), n is the number of classes (here is 34). So, we have to compute the err of each pixel. Iteration of pixels:
     local df_do
     --[[if opt.type ~= 'cuda'  then df_do = torch.Tensor():resizeAs(output) end
     if opt.type == 'cuda' then df_do = torch.CudaTensor():resizeAs(output) end      
        for p = 1, output:size(1) do
          local err = criterion:forward(output[p], targets[i][p])
          f_p = f_p+err
          df_do[p] = criterion:backward(output[p],targets[i][p])
        ----update confusion
        confusion:add(output[p],targets[i][p]) ----in fact, the confusion is based on each pixel
        end
        
        f = f_p/output:size(1) ---- the f of the image is considered as the average of f_p (i.e. the f of each pixel)
         ---- estimate df/dw
        net:backward(inputs[i],df_do)--]]
      local err = criterion:forward(output,targets[i])
      f = f+err
      df_do = criterion:backward(output,targets[i])
      net:backward(inputs[i],df_do)
        ----update confusion
        confusion:batchAdd(output, targets[i])
      end
      gradParameters:div(#inputs)
      f=f/#inputs
      return f, gradParameters
      end
    ----optimize on current mini-batch
    if opt.optimization == 'LBFGS' then
      config = config or {learningRate=opt.learningRate,maxIter = opt.maxIter,nCorrection =29}
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
