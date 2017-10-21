require 'torch'
require 'xlua'
require 'optim'


function test()
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
    local input = testIm[t]
    if opt.type == 'cuda' then input = input:cuda() end
    local target = testLabel[t]
    
    ----test sample
    local pred = net:forward(input)
    --[[for p = 1, pred:size(1) do
      confusion:add(pred[p],target[p])
    end--]]
    confusion:batchAdd(pred,target)
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

