require 'torch'
require 'mattorch'
require 'image'
require 'nn'
require 'optim'

----define the necessary functions 
function buildGraph(mask)
  local  m, n, nodeList, edgeList, count, edge
  nodeList = torch.zeros(mask:sum(),2)
  edge = torch.zeros(2)
  edgeList = {}
  count = 0
  m = mask:size(1) ---the height
  n = mask:size(2)---the width
  for i = 1, m do
    for j = 1, n do
      if mask[i][j]==1 then
        count = count + 1
        nodeList[count][1] = (i-1)*n+j  ---- index of the node
        if i>1 and i < m then
          if j > 1 and j< n then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1] = (i-1)*n + j
            edge[2] = (i-1)*n +(j-1)
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
        elseif j == 1 then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
            end
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end      
        elseif j == n then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j -1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
        end
      elseif i == 1 then
        if j > 1 and j< n then
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1] = (i-1)*n + j
            edge[2] = (i-1)*n +(j-1)
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
      elseif j == 1 then
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end      
      elseif j == n then
          if mask[i+1][j] == 1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j -1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
        end
        
      elseif i == m then
        if j > 1 and j< n then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1] = (i-1)*n + j
            edge[2] = (i-1)*n +(j-1)
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
        elseif j == 1 then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j+1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j + 1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end      
        elseif j == n then
          if mask[i-1][j]==1 then
            edge[1] = (i-1)*n+j
            edge[2] = (i-2)*n +j
            table.insert(edgeList, edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
          if mask[i][j-1]==1 then
            edge[1]=(i-1)*n + j
            edge[2]=(i-1)*n + j -1
            table.insert(edgeList,edge:clone())
            nodeList[count][2]=nodeList[count][2]+1
          end
        end
      end
    end
  end
end

return nodeList, edgeList          
end


function findNode(nodeList, node)
  local pos=nodeList:eq(node):nonzero()
  return pos[1][1]
end


--[[function nodeIdTransfer(nodeList, edgeList)
  local n, m
  n = nodeList:size(1)
  m=#edgeList
  for i = 1, n do
    for j = 1, m do
      if nodeList[i]==edgeList[j][1] then
        edgeList[j][1] = i
      end
      if nodeList[i]==edgeList[j][2] then
        edgeList[j][2]=i
      end
    end
  end
end--]]


function getConnectComponent(nodeList, edgeList)
  local Nedge,Nnode,components,visited,queue, rear, front, v, node, edgeIndxStar, edgeIndxEnd
  components={}
  Nedge = #edgeList
  Nnode=nodeList:size(1)
  visited = torch.zeros(Nnode)
  v = 1
  while true do
  local component ={}  
  visited[v]  = 1
  table.insert(component,nodeList[v][1])
  rear = 0
  front = 0
  rear = rear + 1
  queue = torch.zeros(Nnode) 
  queue[rear] = nodeList[v][1]
  while front<rear do
    front = front + 1
    node = queue [front]
    v = findNode(nodeList, node)
    --print(v)
    --table.insert(component,nodeList[v][1])
    edgeIndxEnd = nodeList[{{1,v},{2}}]:sum()
    edgeIndxStar = edgeIndxEnd-nodeList[v][2] +1
    while edgeIndxStar<= edgeIndxEnd do
      local nodePos = findNode(nodeList, edgeList[edgeIndxStar][2])
      --print()
      --print(nodePos)
      if visited[nodePos] == 0 then
        visited[nodePos] = 1
        table.insert(component,nodeList[nodePos][1])
        rear = rear + 1
        queue[rear] = nodeList[nodePos][1] 
      end
      edgeIndxStar = edgeIndxStar+1
    end
  end
  table.insert(components, torch.Tensor(component):clone())
  if visited:sum()~=Nnode then
    --print('the sum of visited is '..visited:sum())
    local pos = (visited-1):nonzero()
    --print(pos)
    v=pos[1][1]
   -- print('the new v is: '..v)
  else 
    break
  end
  
end
return components
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



function ind2sub(ind, size2d)
  local h, w, tmp, x, y ,pos
  h=size2d[1] -- the height
  w=size2d[2] -- the width
  --print(h)
  --print(w)
  ----the remainer: a%b = a-math.floor(a/b)*b
  tmp = torch.div(ind, w)
  tmp:floor():mul(w)
  x = torch.add(ind, -1,tmp)
  pos=x:eq(0):nonzero()
  if pos:sum()>0 then
    for i = 1,pos:size(1) do
      x[pos[i][1]]=w
    end
end
  y = torch.add(ind,-1,x)
  y:div(w):add(1)
  return x, y
end


function getSourceObjectRegion(sourceLabel,commonObj)
  local region = {}
  local bbox_top = {}
  local bbox_bottom ={}
  local bbox_left = {}
  local bbox_right = {}
  local nodeList, edgeList, components,  maxComp, x, y, px, py,mask, covered, flagLeft,  yLeft, yRight, size, idx -- flagTop m, n,
  --m = sourceLabel:size(1)
  --n = sourceLabel:size(2)
  for obj_I = 1,#commonObj do
    mask = sourceLabel:eq(commonObj[obj_I]):double()
    nodeList, edgeList = buildGraph(mask)
    components = getConnectComponent(nodeList,edgeList)
    ----then, find the biggest component
    size=0
     --local idx
     --print(obj_I)
    for i, comp in ipairs(components) do
      if comp:size(1) >size then
        size = comp:size(1)
        idx = i
        --print(idx)
      end
    end
      --print(idx)
      maxComp = components[idx]:clone()
      ----transfer the index to sub
       --print(maxComp:size())
      x, y = ind2sub(maxComp, mask:size())
      ---- then, find the bbox
      covered = y:max()-y:min()   ---mask[{{y:min(),y:max()},{j}}]:sum()
      --------set the flags
      flagLeft = true
      --flagTop = true
      -------- go through the x coordinate to find the left and right boundaries
      --print(x:min())
      --print(x:max())
      for j = x:min(), x:max() do
        px = x:eq(j)
        py = y:maskedSelect(px) ----note that the sum of px is equal to the elements number of py (i.e. px:sum() = #py)
        --print(mask:size())
        --print(py:min())
        --print(py:max())
        --print(x:min())
        --print(x:max())
       -- print(covered*0.8)
        --print(py:size(1)-covered*0.6)
        --py:max()-py:min() >=py:size(1)
        ----find the left boundry
        if py:size(1) >= 0.8*covered and flagLeft then --and py:size(1) >=(y:max()-y:min())*0.5
          table.insert( bbox_left, j)
          --print(py:size())
          yLeft = py:clone()
          flagLeft = false
          --print(j)
        end
        --print(bbox_left[1])
       -- print(j)
       --print(py:size(1))
       --print(covered-py:size(1))
         --print(obj_I)
         --print(bbox_left[obj_I])
        if not(flagLeft) then
          if (py:size(1) <covered*0.5 and j>bbox_left[obj_I]+50) or j==x:max() then 
            ----print(bbox_left[obj_I]+50)
            table.insert(bbox_right, j-1)
          --print(px:sum())
          --print(py:size())
            yRight = py:clone()
            break
          end
        end
      end
      
       ----check 
      if #bbox_left ==0 or #bbox_right == 0 then
        print('The region shape is not good enough for transfer.')
        return
      end 
      
      --------then find the top and buttom of the box from yLeft and yRight
      table.insert(bbox_top, math.min(yLeft:min(), yRight:min()))
      table.insert(bbox_bottom, math.min(yLeft:max(), yRight:max()))
     end
     
     ----add into region
    region['bbox_top'] = bbox_top
    region['bbox_bottom'] = bbox_bottom 
    region['bbox_left'] = bbox_left
    region['bbox_right'] = bbox_right
   return region  
  end--]]


--[[function getSourceObjectRegion(sourceLabel, commonObj)
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
end--]]


----load data
targetLabelObj = mattorch.load('/home/hogwild/neural-style/images2/target/bird1.jpg.mat')
sourceLabelObj =  mattorch.load('/home/hogwild/neural-style/images2/source/birdoil3.jpg.mat')
targetLabelObj = targetLabelObj.pred:transpose(1,2)
sourceLabelObj = sourceLabelObj.pred:transpose(1,2)
local Target_original_img = image.load('/home/hogwild/neural-style/images2/target/bird1.jpg')
local Source_original_img = image.load('/home/hogwild/neural-style/images2/source/birdoil3.jpg')
----print(targetLabelObj)
----match the objects in 2 images
local NumObj = 21 --the number of the objects containing in the data set
comObj = getCommonObject(targetLabelObj, sourceLabelObj, NumObj)
targetRegion = getTargetObjectRegion(targetLabelObj, comObj)
sourceRegion = getSourceObjectRegion(sourceLabelObj, comObj)
----built the final data for processing
source = {} --{Source_original_img} the background is same to the original image
target = {} --{Target_original_img} the background is same to the original image
print('same objects found: ' , #comObj)
local targetMask, pos, box_left, box_right, box_top, box_bottom, tempSource, insertSource, h,w, m,n 
local maskedTarget --= torch.Tensor():resizeAs(Target_original_img)
for i = 1, #comObj do
  tempSource=Source_original_img[{{},{sourceRegion.bbox_top[i], sourceRegion.bbox_bottom[i]},{sourceRegion.bbox_left[i], sourceRegion.bbox_right[i]}}]
  h = tempSource:size(2)
  w = tempSource:size(3)
  m = targetLabelObj:size(1)
  n = targetLabelObj:size(2)
  if h<0.25*m then
    tempSource=image.scale(tempSource,0.25*m,w,'bilinear')
    h = tempSource:size(2)
  end
  if w<0.25*w then
    tempSource = image.scale(tempSource,h,0.25*n,'billinear')
     w = tempSource:size(3)
  end
  insertSource = torch.repeatTensor(tempSource, 1, math.max(math.floor(m/h),1), math.max(math.floor(n/w),1))
  table.insert(source, insertSource)
  --table.insert(source, Source_original_img[{{},{sourceRegion.bbox_top[i], sourceRegion.bbox_bottom[i]},{sourceRegion.bbox_left[i], sourceRegion.bbox_right[i]}}])
  --print(source[i]:size())
  --table.insert(target, Target_original_img[{{},{targetRegion.bbox_top[i], targetRegion.bbox_bottom[i]},{targetRegion.bbox_left[i], targetRegion.bbox_right[i]}}])
  targetMask = targetLabelObj:eq(comObj[i]):double()
  pos = targetMask:nonzero()
  --targetMask :repeatTensor(targetMask,3,1,1)
  --print(targetMask:size())
  -- print(Target_original_img:size())
  --pos = torch.cmul(Target_original_img, targetMask):
  box_top = pos[{{},{1}}]:min()
  box_bottom = pos[{{},{1}}]:max()
  box_left = pos[{{},{2}}]:min()
  box_right = pos[{{},{2}}]:max()
  maskedTarget = Target_original_img[{{},{box_top,box_bottom},{box_left,box_right}}]
  print(maskedTarget:size())
  table.insert(target,maskedTarget)
  --print(target)
end--]]
--print(#target)

--mask = targetLabelObj:eq(comObj):double
--n,e  = buildGraph(mask)
--comp = getConnectComponent(n,e)

---- style transfer
newimage={}
local loadcaffe_wrap = require 'loadcaffe_wrapper'
local params = {}
--params['image_size'] = 256
params['gpu'] = 0
params['content_weight_bg'] = 1e0
params['content_weight'] = 1e2
params['style_weight'] = 1e2
params['tv_weight'] = 1e-3
params['num_iterations'] = 800
params['normalize_gradients'] = false
params['init'] = 'image'  --'random|image')
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
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    params.backend = 'nn-cpu'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
  end
  
  for  objN = 1, (#comObj) do 
  freeMemory = cutorch.getMemoryUsage(1)
  print('At the beginning of the iteration, the free GPU memory is: '..freeMemory)
  
  local cnn = loadcaffe_wrap.load(params.proto_file, params.model_file, params.backend):float()
  
  --freeMemory = cutorch.getMemoryUsage(1)
 -- print('the free GPU memory is: '..freeMemory)
  
  if params.gpu >= 0 then
    cnn:cuda()
  end
  
  local content_image = target[objN]
  --content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
 --print(content_image_caffe:size())
  
  local style_image = source[objN]
  local dim=torch.zeros(1,2)
  dim[1][1] = content_image:size(2)
  dim[1][2] = content_image:size(3)
  local style_size = math.ceil(params.style_scale * dim:max())  --params.image_size)
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
        if objN == 1 then
           content_weight = params.content_weight_bg
        else
           content_weight = params.content_weight
        end
        
        local loss_module = nn.ContentLoss(content_weight, target, norm):float()
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
 -- freeMemory = cutorch.getMemoryUsage(1)
  --print('the free GPU memory is: '..freeMemory)
  cnn = nil
  collectgarbage()
  -- freeMemory = cutorch.getMemoryUsage(1)
   --print('the free GPU memory is: '..freeMemory)
  
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
content_image_caffe=nil
style_image_caffe = nil
net=nil
gram=nil
collectgarbage()
freeMemory = cutorch.getMemoryUsage(1)
print('after one iteration the free GPU memory is: '..freeMemory)
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
local OBJ, objMask, resultMask, pos, box_bottom, box_top, box_left, box_right, tempim
for i = 2, (#comObj) do
  OBJ = newimage[i]
  tempim = torch.Tensor(result:size())
  objMask = targetLabelObj:eq(comObj[i]):double()
  pos = objMask:nonzero()
  box_top = pos[{{},{1}}]:min()
  box_bottom = pos[{{},{1}}]:max()
  box_left = pos[{{},{2}}]:min()
  box_right = pos[{{},{2}}]:max()
  tempim[{{},{box_top,box_bottom},{box_left,box_right}}] = OBJ
  --print(tempim:size())
  objMask = objMask:repeatTensor(3,1,1)
  resultMask = targetLabelObj:ne(comObj[i]):double()
  resultMask = resultMask:repeatTensor(3,1,1)
  result = torch.cmul(result,resultMask) + torch.cmul(tempim, objMask)
  --print(result:size())
end
image.save('result.png', result)
--]]