require 'torch'


----load the data and the pretrained net
dofile '1_data_preprocess.lua'
dofile '2_load_net.lua'
---- remove the unnecessary layers
net:remove(6) ---- remove 'nn.LogSoftMax' 
--local lastLayer = net:findModules('nn.Linear')
net:remove(5) ----remove the 'nn.Linear'
--net_output = lastLayer[1].gradInput:size(2)

---- generate the features 
--[[trainFeatures = torch.Tensor(trainIm:size(1), lastLayer[1].gradInput:size(1),lastLayer[1].gradInput:size(2))
testFeatures=torch.Tensor(testIm:size(1), lastLayer[1].gradInput:size(1),lastLayer[1].gradInput:size(2) )

for i=1, trainIm:size(1) do
  trainFeature[i] = net:forward(trainIm[i])
end
for i=1, testIm:size(1) do
  testFeature[i] = net:forward(testIm[i])
end

----release the Feature_extr
net = nil
lastLayer = nil
collectgarbage()--]]

