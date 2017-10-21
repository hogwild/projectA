require 'torch'
require 'xlua'
require 'optim'


if opt.type == 'cuda' then
  Descriptor:float()
  criterion:float()
end

print '==> defining training procedure'
--classes
classes = {'1','5',	'6',	'7',	'9',	'10',	'12',	'13',	'14',	'15',	'16',	'18',	'19',	'20',	'21',	'22',	'23',	'24',	'25',	'26',	'28',	'29',	'32',	'33',	'34'}
--cofusion matrix
confusion = optim.ConfusionMatrix(classes)
--log results to files
trainLogger = optim.Logger(paths.concat(opt.save,'train_discriptor.log'))
testLogger = optim.Logger(paths.concat(opt.save,'test_discriptor.log'))

if Descriptor then
   parameters, gradParameters = Descriptor:getParameters()
end

local trsize = trainIm:size(1)
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
    
    ----Switch two nets between GPU memory and CPU memory, because GPU memory is not enough
    --Descriptor:float() 
    --net:cuda() 
    for i = t, math.min(t+opt.batchSize-1,trainIm:size(1)) do
      local input = net:forward(trainIm[shuffle[i]]:cuda())
      local target = trainLabel[shuffle[i]]:view(-1)
      if opt.type == 'cuda' then target = target:float() end
      if opt.type == 'cuda' then input = input:float() end
      table.insert(inputs,input)
      table.insert(targets,target)
    end
    --net:float() 
    --Descriptor:cuda()  
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
        local output = Descriptor:forward(inputs[i]) 
     local df_do
      local err = criterion:forward(output,targets[i])
      f = f+err
      df_do = criterion:backward(output,targets[i])
      Descriptor:backward(inputs[i],df_do)
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
local filename = paths.concat(opt.save, 'Superpixel_Descriptor.net')
os.execute('mkdir -p'..sys.dirname(filename))
print('==>saving net to '..filename)
torch.save(filename, Descriptor)
----next epoch
epoch = epoch+1
end
