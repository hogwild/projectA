require 'torch'
require 'image'
require 'nnx'
require 'mattorch'
require 'xlua'


--- load data
local trainImList = {}
local trainLabList={}
local testImList = {}
local testLabList={}
local i=1
local j=1
local trainim_list = io.open('/home/hogwild/Documents/NatureSence_Dataset/opencountry_trainimg.txt')
local trainlab_list = io.open('/home/hogwild/Documents/NatureSence_Dataset/opencountry_trainlab.txt')
local testim_list = io.open('/home/hogwild/Documents/NatureSence_Dataset/opencountry_testimg.txt')
local testlab_list = io.open('/home/hogwild/Documents/NatureSence_Dataset/opencountry_testlab.txt')
print '==> load the name list '
while true do
  local temp_im = trainim_list:read()
  local temp_lab = trainlab_list:read()
  if temp_im == nil then
    collectgarbage()
    trainim_list:close()
    trainlab_list:close()
    print("\n==> train name list loaded " .. (i-1))
    break
  else
    trainImList[i] = temp_im
    trainLabList[i] =temp_lab
    i = i+1
  end
end

while true do
  local temp_im = testim_list:read() 
  local temp_lab = testlab_list:read()
  if temp_im == nil then
    testim_list:close()
    testlab_list:close()
    print('\n==> test name list loaded '..(j-1))
    break
  else
    testImList[j]=temp_im
    testLabList[j]=temp_lab
    j = j+1
  end
end
local path_im = '/home/hogwild/Documents/NatureSence_Dataset/images/'
local path_label='/home/hogwild/Documents/NatureSence_Dataset/opencountry_labels/'
 trainIm = torch.Tensor(#trainImList,3,256,256)
 testIm = torch.Tensor(#testImList,3,256,256)
 trainLabel = torch.Tensor(#trainLabList,1,256,256)
 testLabel = torch.Tensor(#testLabList,1,256,256)
for i = 1,#trainImList do
  trainIm[i] = image.load(path_im..trainImList[i])
  local temp= mattorch.load(path_label..trainLabList[i])  ----the label information is store in a matlab mat file
  trainLabel[i] = temp.S:transpose(1,2) ---- Matlab uses a column-major representation, Torch is row-major, so we have to transpose the data, the labels are in the existed Matlab mat files.
end
for i = 1, #testImList do
  testIm[i] = image.load(path_im..testImList[i])
  local temp = mattorch.load(path_label..testLabList[i])
  testLabel[i] = temp.S:transpose(1,2) ----for the same reason as the train data
end
----to reshape the labels in to 'one image one vector' 
trainLabel = torch.reshape(trainLabel,trainLabel:size(1),trainLabel:size(3)*trainLabel:size(4))
testLabel = torch.reshape(testLabel,testLabel:size(1),testLabel:size(3)*testLabel:size(4))
---- the labels should begin from 1, but in the input 'mat' file, it begins from 0, so, we add 1 to the labels
trainLabel = trainLabel + 1
testLabel = testLabel + 1
----translate the data into float for saving memory
trainIm=trainIm:float()
testIm=testIm:float()
trainLabel=trainLabel:float()
testLabel=testLabel:float()
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
  mean[i] = trainIm[{{},i,{},{}}]:mean()
  std[i]=trainIm[{{},i,{},{}}]:std()
  trainIm[{{},i,{},{}}]:add(-mean[i])
  trainIm[{{},i,{},{}}]:div(std[i])
end
torch.save('/home/hogwild/neural-style/results/mean.dat', mean)
torch.save('/home/hogwild/neural-style/results/std.dat', std)
for i , channel in ipairs(channels) do
  testIm[{{},i,{},{}}]:add(-mean[i])
  testIm[{{},i,{},{}}]:div(std[i])
end
---- to normalizaion locally on Y channel
print '==>preprocessing data: normalize Y (luminance) channel locally'
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):float()
for i = 1, trainIm:size(1) do 
  trainIm[{i,{1},{},{}}] = normalization(trainIm[{i,{1},{},{}}])
end
for i = 1, testIm:size(1) do
  testIm[{i,{1},{},{}}] = normalization(testIm[{i,{1},{},{}}])
end
print '==>preprocessing data: normalize U channel locally'
for i = 1, trainIm:size(1) do 
  trainIm[{i,{1},{},{}}] = normalization(trainIm[{i,{2},{},{}}])
end
for i = 1, testIm:size(1) do
  testIm[{i,{1},{},{}}] = normalization(testIm[{i,{2},{},{}}])
end
print '==>preprocessing data: normalize V channel locally'
for i = 1, trainIm:size(1) do 
  trainIm[{i,{1},{},{}}] = normalization(trainIm[{i,{3},{},{}}])
end
for i = 1, testIm:size(1) do
  testIm[{i,{1},{},{}}] = normalization(testIm[{i,{3},{},{}}])
end

----to verify that data is properly normalized:
for i, channel in ipairs(channels) do
  trainMean = trainIm[{{},i}]:mean()
  trainStd = trainIm[{{},i}]:std()
  testMean = testIm[{{},i}]:mean()
  testStd = testIm[{{},i}]:std()
  print('training data, '..channel..'-channel, mean: '..trainMean)
  print('training data, '..channel..'-channel, standard deviation: '..trainStd)
  print('test data, '..channel..'-channel, mean: '..testMean)
  print('test data, '..channel..'-channel, standard deviation: '..testStd)
end