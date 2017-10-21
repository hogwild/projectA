require 'torch'
require 'imgraph'
require 'gm'
require 'image'
require 'cunn'
require 'nnx'
require 'optim'


---- define the functions
function labelbySuperpixelFeat(img, D_i, path)
  --local inputimg = image.convolve(img, image.gaussian(5), 'same')
  local graph = imgraph.graph(img)
  --local graph = imgraph.graph(inputimg)
  local segm = imgraph.segmentmst(graph)
  local components=imgraph.extractcomponents(segm,D_i,'masked')
  print('the superpixel number is ', components:size())
 
---- step2. label based on superpixels
  local label = torch.Tensor(components:size())
  local labeled_img = torch.zeros(img:size(2),img:size(3))
  local d_k
  ----print(labeled_img:size())
  for i = 1, components:size() do
  --local masks = torch.repeatTensor(components.mask[i],components.patch[i]:size(1),1,1)
  --local d_k = torch.cmul(components.patch[i], masks):sum(2):sum(3):view(-1)
     d_k=components.patch[i]:sum(2):sum(3):view(-1)
     d_k:div(components.surface[i])
     m, label[i] = torch.max(d_k,1)
     m=nil
     labeled_img[{{components.bbox_top[i],components.bbox_bottom[i]},{components.bbox_left[i],   components.bbox_right[i]}}]:add(components.mask[i]*label[i])
   end
   local mstsegmcolor = imgraph.colorize(segm)
   local colored_label, colormap 
   colored_label, colormap = imgraph.colorize(labeled_img)  ----visualize the label result
   --print(colormap:max(), colormap:min())
   image.savePNG(path, colored_label)
   image.savePNG('/home/hogwild/neural-style/superpixel.png', mstsegmcolor)
   return label, components
end


function getObjectLabel(components, labelSuperpixel, original_img)
  local labelObject = torch.zeros(original_img:size(2),original_img:size(3))
  for i = 1, components:size() do
    labelObject[{{components.bbox_top[i],components.bbox_bottom[i]},{components.bbox_left[i],   components.bbox_right[i]}}]:add(components.mask[i]*labelSuperpixel[i])
  end
    return labelObject
end


function getCommonObject(targetLabelObject, sourceLabelObject, Nobject, threshold )
  local objInTarget = torch.histc(targetLabelObject, Nobject,1,Nobject)
  local objInSource = torch.histc(sourceLabelObject, Nobject,1,Nobject)
  local commonObj={}
  local T = threshold or 600
  for i = 1, Nobject do
    if objInSource[i] > T and objInTarget[i] > T then
      table.insert(commonObj,i)
    end
  end
  return commonObj
end

function getTargetObjectRegion(targetLabel, commonObj)
  local region = {}
  local bbox_top = {}
  local bbox_bottom = {}
  local bbox_left = {}
  local bbox_right = {}
  local mask, pos
  for i = 1, #commonObj do
    mask = targetLabel:eq(commonObj[i]):double()
    pos=mask:nonzero()
    table.insert(bbox_top, pos[{{},{1}}]:min())
    table.insert(bbox_bottom, pos[{{},{1}}]:max())
    table.insert(bbox_left, pos[{{},{2}}]:min())
    table.insert(bbox_right,pos[{{},{2}}]:max())
  end
  region['bbox_top'] = bbox_top
  region['bbox_bottom'] = bbox_bottom
  region['bbox_left'] = bbox_left
  region['bbox_right'] = bbox_right
  return region
end


function getSourceObjectRegion(sourceLabel, commonObj)
  local region = {}
  local bbox_top = {}
  local bbox_bottom ={}
  local bbox_left = {}
  local bbox_right = {}
  local mask, pos
  for i = 1, #commonObj do
    mask =  sourceLabel:eq(commonObj[i]):double()
    pos = mask:nonzero() ----pos is a nx2 tensor, pos[{{},{1}}] is the coordinate of y , pos[{{},{2}}] is the coordinate of x
    local Start_y = {pos[1][1]}
    local End_y = {}
    local gap_y = {}
    --local count = 1
    local breakpoint = {1}  ---- NOTE: an extrem case, that the gap only contain one point !!!
    ---- find the gap in y direction
    for j = 2, pos:size(1) do 
      if (pos[j][1] - pos[j-1][1]) > 1 then
        table.insert(End_y, pos[j-1][1])
        table.insert(Start_y, pos[j][1])
        --count = count+1
        table.insert(breakpoint, j)
      end
      if j == pos:size(1) then
        table.insert(End_y, pos[j][1]) 
        table.insert(breakpoint, j)
      end
    end
    gap_y['Start'] = Start_y
    gap_y['End'] = End_y 
    
    local gap_x = {}
     ----count = 1----reset count 
    for p = 2, #breakpoint do
      local Start_x = {}
      local End_x = {}
      local gapIny = {}
      local elem_x = torch.Tensor(breakpoint[p]-breakpoint[p-1])
      --print(breakpoint)
      --print(elem_x:size())
      elem_x = pos[{{breakpoint[p-1], breakpoint[p]-1},{2}}]:sort(1):view(-1)
      --print (elem_x)
      --print(#elem_x)
      table.insert(Start_x,elem_x[1])
      --print(Start_x)
      for q = 2, elem_x:size(1) do
        if (elem_x[q] - elem_x[q-1]) > 1 then
          table.insert(End_x, elem_x[q-1])
          table.insert(Start_x, elem_x[q])
       --  print(Start_x)
        end
        if q== elem_x:size(1) then
          table.insert(End_x, elem_x[q])
         -- print(End_x)
        end
      end
      gapIny['xStartIny'] = Start_x
      gapIny['xEndIny'] = End_x
      --print(gapIny)
      table.insert( gap_x, gapIny)  ---- the element in gap_x[p-1] is a table corresponding to (gap_y.Start[p-1],gap_y.End[p-1])  
    end
   -- print(gap_x)
    ----find the biggest region
    local area = 0
    local areaTmp = 0
    local best_bottom,best_top,best_left,best_right
    local top, bottom, left, right
    for m = 1, #gap_y.Start do
      top = gap_y.Start[m]
      bottom = gap_y.End[m]
      for n = 1, #gap_x[m].xStartIny do
        left = gap_x[m].xStartIny[n] 
        right = gap_x[m].xEndIny[n]
        areaTmp = mask[{{top, bottom},{left, right}}]:sum()
        if areaTmp>area then 
          area = areaTmp
          best_top = top
          best_bottom = bottom
          best_left = left
          best_right = right
        end
      end
    end
    table.insert(bbox_top, best_top)
    table.insert(bbox_bottom,best_bottom)
    table.insert(bbox_left,best_left)
    table.insert(bbox_right,best_right)
    end
    region['bbox_top'] = bbox_top
    region['bbox_bottom'] = bbox_bottom 
    region['bbox_left'] = bbox_left
    region['bbox_right'] = bbox_right
    return region
end



--[[function crfInference(img, D_i, path)
  local graph = imgraph.graph(img)
  local segm = imgraph.segmentmst(graph)
  local gradient = imgraph.graph2map(graph)
  local components = imgraph.extractcomponents(segm, D_i, 'masked')
  local alpha = 0.1
  local beta = 20
  local gama = 200
  imgraph.adjacency(segm,components);
  local nNode = components:size()
  local nStates = D_i:size(2)
  local g = gm.graph{adjacency = components.adjacency, nStates=nStates,  maxIter=10, verbose=true}
end--]]

---- step1. generate features for every pixel
----preprocess images
Source_original_img = image.loadJPG('/home/hogwild/Pictures/monet_tree.jpg')
Source_original_img = image.scale(Source_original_img,256,256,'bilinear')
Target_original_img = image.loadJPG('/home/hogwild/Documents/NatureSence_Dataset/images/opencountry_sopen11.jpg')
--Target_original_img = image.scale(Target_original_img,256,256,'bilinear')
--print(original_img:size())
--source_img
----rgb2yuv
print '==>preprocessing data: colorspace RGB -> YUV'
target_img = image.rgb2yuv(Target_original_img)
source_img = image.rgb2yuv(Source_original_img)
---- normalizeation
channels = {'y','u','v'}
print '==>preprocessing data: normalize each feature (channel) globally'
mean = torch.load('/home/hogwild/neural-style/results/mean.dat')
std = torch.load('/home/hogwild/neural-style/results/std.dat')
for i, channel in ipairs(channels) do
  target_img[{i,{},{}}]:add(-mean[i])
  target_img[{i,{},{}}]:div(std[i])
  source_img[{i,{},{}}]:add(-mean[i])
  source_img[{i,{},{}}]:div(std[i])
end
--print(target_img)
print '==>preprocessing data: normalize Y (luminance) channel locally'
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1)--:float()
target_img[{1, {}, {}}] = normalization(target_img[{1, {}, {}}]:resize(1,target_img:size(2),target_img:size(3)))
source_img[{1, {}, {}}] = normalization(source_img[{1, {}, {}}]:resize(1,source_img:size(2),source_img:size(3)))
print '==>preprocessing data: normalize U channel locally'
target_img[{2, {}, {}}] = normalization(target_img[{2, {}, {}}]:resize(1,target_img:size(2),target_img:size(3)))
source_img[{2, {}, {}}] = normalization(source_img[{2, {}, {}}]:resize(1,source_img:size(2),source_img:size(3)))
print '==>preprocessing data: normalize V channel locally'
target_img[{3, {}, {}}] = normalization(target_img[{3, {}, {}}]:resize(1,target_img:size(2),target_img:size(3)))
source_img[{3, {}, {}}] = normalization(source_img[{3, {}, {}}]:resize(1,source_img:size(2),source_img:size(3)))

---- extractor scale invariant features
dofile '2_load_net.lua' -- LOAD : net
net:remove(6)  --remove the unnecessary layers
net:remove(5)
--featureExtractNet = torch.load('/home/hogwild/neural-style/results/FeatureExtractor.net')
--Descriptor=torch.load('/home/hogwild/neural-style/results/Superpixel_Descriptor.net')
print(net)
targetFeat = net:forward(target_img:cuda())
targetFeat=targetFeat:float()
sourceFeat = net:forward(source_img:cuda())
sourceFeat=sourceFeat:float()
net=nil  ---- clean net to save GPU memory
collectgarbage()
--print(targetFeat)--]]
dofile '2_load_descriptor.lua' --- LOAD: descriptor
print(descriptor)
--d=descriptor
---descriptor:cuda()

----extractor superpixel based features
targetD_i = descriptor:forward(targetFeat)
targetD_i:resize(targetD_i:size(2),target_img:size(2),target_img:size(3))
targetD_i=targetD_i:double()
sourceD_i = descriptor:forward(sourceFeat)
sourceD_i:resize(sourceD_i:size(2),source_img:size(2),source_img:size(3))
sourceD_i=sourceD_i:double()
descriptor=nil ---- clearn descriptor to save GPU memory
collectgarbage()
--print(targetD_i:size())
--print(targetD_i:type())
target_path_save = '/home/hogwild/neural-style/target_label_result.png'
targetLabel , targetComponents = labelbySuperpixelFeat(Target_original_img, targetD_i, target_path_save)
targetLabelObj = getObjectLabel(targetComponents,targetLabel,Target_original_img)

source_path_save = '/home/hogwild/neural-style/source_label_result.png'
sourceLabel , sourceComponents = labelbySuperpixelFeat(Source_original_img, targetD_i, source_path_save)
sourceLabelObj = getObjectLabel(sourceComponents,sourceLabel,Source_original_img)

----match the objects in 2 images
comObj = getCommonObject(targetLabelObj, sourceLabelObj, 25)

targetRegion = getTargetObjectRegion(targetLabelObj, comObj)
sourceRegion = getSourceObjectRegion(sourceLabelObj, comObj)

source = {Source_original_img}
target = {Target_original_img}
print('same objects found: ' , #comObj)
local targetMask
local maskedTarget = torch.Tensor():resizeAs(Target_original_img)
for i = 1, #comObj do
  table.insert(source, Source_original_img[{{},{sourceRegion.bbox_top[i], sourceRegion.bbox_bottom[i]},{sourceRegion.bbox_left[i], sourceRegion.bbox_right[i]}}])
  --table.insert(target, Target_original_img[{{},{targetRegion.bbox_top[i], targetRegion.bbox_bottom[i]},{targetRegion.bbox_left[i], targetRegion.bbox_right[i]}}])
  targetMask = targetLabelObj:eq(comObj[i]):double()
  targetMask :repeatTensor(targetMask,3,1,1)
  print(targetMask:size())
  -- print(Target_original_img:size())
  maskedTarget = torch.cmul(Target_original_img, targetMask)
  --print(maskedTarget)
  table.insert(target,maskedTarget)
  --print(target)
end
--print(target)

--[[local sourceD_i = descriptor:forward(sourceFeat)
local path = '/home/hogwild/source_label_result.png'
sourceLabel, sourceComponents = labelbySuperpixelFeat(source_img, sourceD_i, path)--]]





---- step3. set the conditional random field and labeling




---- step5.  transfer style between same objects--]]
newimage={}
local loadcaffe_wrap = require 'loadcaffe_wrapper'
local params = {}
params['image_size'] = 256
params['gpu'] = 0
params['content_weight'] = 5e0
params['style_weight'] = 1e2
params['tv_weight'] = 1e-3
params['num_iterations'] = 800
params['normalize_gradients'] = false
params['init'] = 'random'
params['optimizer'] = 'lbfgs'
params['learning_rate'] = 1e1
params['print_iter'] = 50
params['save_iter'] = 100
params['output_image'] = 'out.png'
params['style_scale'] = 1.0
params['pooling'] = 'max'
params['proto_file'] = 'models/VGG_ILSVRC_19_layers_deploy.prototxt'
params['model_file'] = 'models/VGG_ILSVRC_19_layers.caffemodel'
params['backend'] = 'cudnn'

local function main(params)
  if params.gpu >= 0 then
    --require 'cutorch'
    --require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    params.backend = 'nn-cpu'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
  end
   for  objN = 1, (#comObj+1) do 
  local cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()
  if params.gpu >= 0 then
    cnn:cuda()
  end
  local content_image = target[objN]
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_image = source[objN]
  local style_size = math.ceil(params.style_scale * params.image_size)
  style_image = image.scale(style_image, style_size, 'bilinear')
  local style_image_caffe = preprocess(style_image):float()
  
  if params.gpu >= 0 then
    content_image_caffe = content_image_caffe:cuda()
    style_image_caffe = style_image_caffe:cuda()
  end
  
  -- Hardcode these for now
  local content_layers = {23}
  local style_layers = {2, 7, 12, 21, 30}
  local style_layer_weights = {1e0, 1e0, 1e0, 1e0, 1e0}

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      tv_mod:cuda()
    end
    net:add(tv_mod)
  end
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then avg_pool_layer:cuda() end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if i == content_layers[next_content_idx] then
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if i == style_layers[next_style_idx] then
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          gram = gram:cuda()
        end
        local target_features = net:forward(style_image_caffe):clone()
        local target = gram:forward(target_features)
        target:div(target_features:nElement())
        local weight = params.style_weight * style_layer_weights[next_style_idx]
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(weight, target, norm):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  collectgarbage()
  
  -- Initialize the image
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    img = img:cuda()
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
        table.insert(newimage,disp)
      end
      image.save(filename, disp)
      
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:backward(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end  
end


function build_filename(output_image, iteration)
  local idx = string.find(output_image, '%.')
  local basename = string.sub(output_image, 1, idx - 1)
  local ext = string.sub(output_image, idx)
  return string.format('%s_%d%s', basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

main(params)

----merg the parts
result = newimage[1]
local OBJ, objMask, resultMask
for i = 2, (#comObj+1) do
  OBJ = newimage[i]
  objMask = targetLabelObj:eq(comObj[i-1]):double()
  objMask = objMask:repeatTensor(3,1,1)
  resultMask = targetLabelObj:ne(comObj[i-1]):double()
  resultMask = resultMask:repeatTensor(3,1,1)
  result = torch.cmul(result,resultMask) + torch.cmul(OBJ, objMask)
  end
  
