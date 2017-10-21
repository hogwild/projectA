require 'torch'
require 'nnx'
require 'image'
--require 'gm'
--require 'imgraph'
require 'optim'
require 'mattorch'


local function main(params)
---- set parameters
local ratios={1,2,4}
local kW=0
local kH=0
local dW=0
local dH=0
local xDimIn
local yDimIn
local xDimOut
local yDimOut
local nChannels = 3
local Nstates = {16,64,256}
local fanIn ={1,8,32}
local filtSize = 7
local poolSize = 2
local processors = buildProcessor(#ratios, nChannels, Nstates, fanIn, filtSize, poolSize)
----build the net
local net = nn.Sequential()
local Pyramid = nn.SpatialPyramid(ratios, processors, kW, kH, dW, dH, xDimIn, yDimIn, xDimOut, yDimOut) 
net:add(Pyramid)
net:add(nn.logSoftMax())
criterion=nn.ClassNLLCriterion()

--- load data
local trainImList = {}
local trainLabList={}
local testImList = {}
local testLabList={}
local i=1
local j=1
trainim_list = io.open('/home/hogwild/Documents/SiftFlowDataset/trainimgs.txt ')
trainlab_list = io.open('/home/hogwild/Documents/SiftFlowDataset/trainlabels.txt')
testim_list = io.open('/home/hogwild/Documents/SiftFlowDataset/testimgs.txt')
testlab_list = io.open('/home/hogwild/Documents/SiftFlowDataset/testlabels.txt')
while true do
  local temp_im = trainim_list:read()
  local temp_lab = trainlab_list:read()
  if temp_im == '' then
    trainim_list:close()
    trainlab_list:close()
    break
  else
    trainImList[i] = temp_im
    trainLabList[i] =temp_lab
    i = i+1
  end
while true do
  local temp_im = testim_list:read() 
  local temp_lab = testlab_list:read()
  if temp_im == '' then
    testim_list:close()
    testlab_list:close()
    break
  else
    testImList[j]=temp_im
    testLabList[i]=temp_lab
    j = j+1
  end
  
local path_im = '/home/hogwild/Documents/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories/'
local path_lable='/home/hogwild/Documents/SiftFlowDataset/SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories/'
local trainIm = torch.Tensor(#trainImList,3,256,256)
local testIm = torch.Tensor(#testImListist,3,256,256)
local trainLabel = torch.Tensor(#trainLabList,1,256,256)
local testLabel = torch.Tensor(#testLabList,1,256,256)
for i = 1,#trainImList do
  trainIm[i] = image.load(path_im..trainImList[i])
  trainLabel[i] = mattorch.load(path_label..trainLabList[i])  ----the label information is store in a matlab mat file
  trainLabel[i] = trainLabel[i].S:transpose(1,2) ---- Matlab uses a column-major representation, Torch is row-major, so we have to transpose the data
end
for 1 = 1, #testImList do
  testIm[i] = image.load(path_im..testImList[i])
  testLabel[i] = mattorch.load(path_label..testLabList[i])
  testLabel[i] = testLabel[i].S:transpose(1,2) ----for the same reason as the train data
end
---- transfer the RGB to YUV
print '==>preprocessing data: colorspace RGB -> YUV'
for i = 1, trainIm:size(1) do 
  trainIm[i] = image.rgb2yuv(trainIm[i])
end
for i = 1, testIm:size(1) do
  testIm[i] = image.rgb2yuv(testIm[i])
end
----preprocessing the images, normalize the dataset
channels = {'y','u','v'}
print '==>preprocessing data: normalize each feature (channel) globally'
mean={}
std={}
for i, channel in ipairs(channels) do
  mean[i] = trainIm[{{},[i],{},{}}]:mean()
  std[i]=trainIm[{{},[i],{},{}}]:std()
  trainIm[{{},i,{},{}}]:add(-mean[i])
  trainIm[{{},i,{},{}}]:div(std[i])
end
for i , channels in ipairs(channels) do
  testIm[{{},i,{},{}}]:add(-mean[i])
  testIm[{{},i,{},{}}]:div(std[i])
end
---- to normalizaion locally on Y channel
print '==>preprocessing data: normalize Y (luminance) channel locally'
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood):float()
for i = 1, trainIm:size(1) do 
  trainIm[{i,{1},{},{}}] = normalization(trainIm[{i,{1},{},{}}])
end
for i = 1, testIm:size(1) do
  testIm[{i,{1},{},{}}] = normalization(testIm[{i,{1},{},{}}])
end
----to verify that data is properly normalized:
for i, channels in ipairs(channels) do
  trainMean = trainIm[{{},i}]:mean()
  trainStd = trainIm[{{},i}]:std()
  testMean = testIm[{{},i}]:mean()
  testStd = testIm[{{},i}]:std()
  print('training data, '..channel..'-channel, mean: '..trainMean)
  print('training data, '..channel..'-channel, standard deviation: '..trainStd)
  print('test data, '..channel..'-channel, mean: '..testMean)
  print('test data, '..channel..'-channel, standard deviation: '..testStd)
end
----training 
--classes
classes = {'1','2','3','4','5','6','7','8','9','10','11','12', '13','14' ,'15', '16', '17', '18', '19', '20', '21','22','23','24', '25', '26','27','28','29','30','31','32','33','0'}
--cofusion matrix
confusion = optim.ConfusionMatrix(classes)
--log results to files
trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

----retrive parameters and gradients:
if net then
  parameters, gradParameters = net:getParameters()
end
----training function
local function train()
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
      local input = trainIm[shuffle[i]]:double()
      local target = trainLabel[shuffle[i]]
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
      ----f is the average of all criterions
      local f = 0
      ----evaluation funtion for complete mini patch
      for i = 1, #inputs do
        ----estimate f
        local output = net:forward(inputs[i])
        local err = criterion:forward(output, targets[i])
        f = f+err
        ---- estimate df/dw
        local df_do = criterion:backward(output,targets[i])
        net:backward(inputs[i],df_do)
        ----update confusion
        confusion:add(output,targets[i])
      end
      gradParameters:div(#inputs)
      f=f/#inputs
      return f, gradParameters
    end
    ----optimize on current mini-batch
    if opt.optimization == 'LBFGS' then
      config = config or {learningRate=opt.learningRate,maxIter = opt.maxIter,nCorrection =34}
      optim.lbfgs(feval,parameters,config)
    else opt.optimization =='SGD' then
      config=config or {learningRate = opt.learningRate, weightDecay = opt.weightDecay, momentum = opt.momentum, learningRateDecay = 5e-7}
      optim.sgd(feval, parameters,config)
    end
  end
  
  ----time taken
  time = sys.clock()-time
  time = time/trainIm:size()
  print('==>time to learn 1 sample = '..(time*1000)..'ms')
  ---print confusion matrix
  print(confusion)
  confusion:zero()
  ----update logger/plot
  trainLogger:add{['%mean class accuracy(train set)']=confusion.totalValid*100}
  if opt.plot then
    trainLogger:style{'%mean class accuracy(train set)'='-'}
    trainLogger:plot()
  end
----save/log current net
local filename = paths.concat(opt.save, 'FeatExtractor.net')
os.execute('mkdir -p'..sys.dirname(filename))
print('==>saving net to '..filename)
torch.save(filename, net)
----next epoch
epoch = epoch+1
end
------test over test data

local function test()
  local time = sys.clock()
  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end
  ----test over test data
  print('==>testing on test set: ')
  for t = 1, testIm:size(1) do
    ----disp progress
    xlua.progress(t,testIm:size(1))
    ----get new sample
    local input = testIm[t]:double()
    local target = testLabel[t]
    
    ----test sample
    local pred = net:forward(input)
    confusion:add(pred,target)
  end
  ----timing
   time = sys.clock() - time
   time = time/testIm:size(1)
   print('==>time to test 1 sample =  '..(time*1000)..'ms')
   ----print confusion matrix
   print(confusion)
   confusion:zero()
   
   ----update log/plot
   testLogger:add{['%mean class acccuracy (test set)']=confusion.totalValid*100}
   if opt.plot then
     testLogger:style{['%mean class accuracy(test set)']='-'}
     testLogger:plot()
   end
   -----average param use?
   if average then
     ---restore parameters
     parameters:copy(cachedparams)
   end
 end 
  
    while epoch <2
    train()
    test()
  end
  
end


----the ConvNet 
function buildProcessor(nProcessors,nChannels,Nstates,fanIn,filtSize,poolSize)
  local processors={}
  local nfeats=nChannels
  local Np = nProcessors
  local nstates = Nstates
  local fanin = fanIn
  local filtsize =filtSize
  local poolsize = poolSize
  local model
  for i = 1, Np do
    if i==1 then
    --stage1:covolution layer
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nfeats,nstates[1],fanin[1]),filtsize,filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --This MaxPooling can be substituted by a LPPooling + SubtractiveNormalization
    --stage2: covolution layer
    model: add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1],nstates[2],fanin[2]),filtsize,filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --same to stage1
  --stage3: a simple filter bank
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[2],nstates[3],fanin[3]),filtsize,filtsize))
  --- add the model to processors
    processors[i]=nn.Sequential()
    processors[i]:add(model)
    else
    processors[i]=processors[1]:clone('weight','bias','gradWeight','gradBias')
    end
  end
  return processors
end

