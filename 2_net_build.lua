require 'torch'
require 'image'
require 'nnx'


---- set parameters
local ratios={1,2,4}
local imgwidth = 256
local imgheight = 256
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
local nClass = 25

----the ConvNet 
local function buildProcessor(nProcessors,nChannels,Nstates,fanIn,filtSize,poolSize)
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
    --model:add(nn.SpatialConvolutionMap(nn.tables.random(nfeats,nstates[1],fanin[1]),filtsize,filtsize))
    model:add(nn.SpatialConvolutionMM(nfeats,nstates[1],filtsize,filtsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --This MaxPooling can be substituted by a LPPooling + SubtractiveNormalization
    --stage2: covolution layer
    --model: add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1],nstates[2],fanin[2]),filtsize,filtsize))
    model: add(nn.SpatialConvolutionMM(nstates[1],nstates[2],filtsize,filtsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --same to stage1
  --stage3: a simple filter bank
    --model:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[2],nstates[3],fanin[3]),filtsize,filtsize))
    model:add(nn.SpatialConvolutionMM(nstates[2],nstates[3],filtsize,filtsize))
  --- add the model to processors
    --processors[i]=nn.Sequential()
    processors[i]=model:clone() ---- ? '=' or model:clone()  need to be verified!
    else
    processors[i]=processors[1]:clone('weight','bias','gradWeight','gradBias')
    end
  end
  return processors
end
---- create the processors
local processors = buildProcessor(#ratios, nChannels, Nstates, fanIn, filtSize, poolSize)

---- the spatialpyramid: adjusted from the one in nnx package, to make the outputs in the same size of the input image
local SpatialPyramid2, parent = torch.class('nn.SpatialPyramid2', 'nn.Module')

function SpatialPyramid2:__init(ratios, processors, imgH, imgW, kW, kH, dW, dH, xDimIn, yDimIn,
			       xDimOut, yDimOut, prescaled_input)
   parent.__init(self)
   self.prescaled_input = prescaled_input or false
   assert(#ratios == #processors)
   
   self.ratios = ratios
   self.kH = kH or 0
   self.kW = kW or 0
   self.dH = dH or 0
   self.dW = dW or 0
   self.imgW = imgW
   self.imgH = imgH
   self.focused = false
   self.x = 0
   self.y = 0
   self.wFocus = 0
   self.hFocus = 0
   self.processors = processors

   local wPad = kW-dW
   local hPad = kH-dH
   local padLeft   = math.floor(wPad/2)
   local padRight  = math.ceil (wPad/2)
   local padTop    = math.floor(hPad/2)
   local padBottom = math.ceil (hPad/2)

   -- focused
   self.focused_pipeline = nn.ConcatTable()
   for i = 1,#self.ratios do
      local seq = nn.Sequential()
      seq:add(nn.SpatialPadding(0,0,0,0, yDimIn, xDimIn))
      --[[seq:add(nn.SpatialReSamplingEx{rwidth=1.0/self.ratios[i], rheight=1.0/self.ratios[i],
				     xDim = xDimIn, yDim = yDimIn, mode='average'})--]]
    if opt.type=='cuda' then
     seq:add(nn.SpatialSamplingBilinear({rwidth=1.0/self.ratios[i],rheight=1.0/self.ratios[i]}))
   else
     seq:add(nn.SpatialReSampling({rwidth=1.0/self.ratios[i],rheight=1.0/self.ratios[i]}))
      --seq:add(nn.SpatialSubSampling(3,1,1,1.0/self.ratios[i],1.0/self.ratios[i]))
      end
      seq:add(processors[i])
      self.focused_pipeline:add(seq)
   end

   -- unfocused
   if prescaled_input then
      self.unfocused_pipeline = nn.ParallelTable()
   else
      self.unfocused_pipeline = nn.ConcatTable()
   end
   for i = 1,#self.ratios do
      local seq = nn.Sequential()
      if not prescaled_input then
	 --[[seq:add(nn.SpatialReSamplingEx{rwidth=1.0/self.ratios[i], rheight=1.0/self.ratios[i],
					xDim = xDimIn, yDim = yDimIn, mode='average'})--]]
  if opt.type=='cuda' then
   seq:add(nn.SpatialSamplingBilinear({rwidth=1.0/self.ratios[i],rheight=1.0/self.ratios[i]}))
 else
   seq:add(nn.SpatialReSampling({rwidth=1.0/self.ratios[i],rheight=1.0/self.ratios[i]}))
  -- seq:add(nn.SpatialSubSampling(3,1,1,self.ratios[i],self.ratios[i]))
   --end
	 --seq:add(nn.SpatialPadding(padLeft, padRight, padTop, padBottom, yDimIn, xDimIn))
  end
    seq:add(processors[i])
    --[[seq:add(nn.SpatialReSamplingEx{owidth=self.imgW, oheight=self.imgH,
				     xDim=xDimOut, yDim=yDimOut, mode='simple'})--]]
    if opt.type=='cuda' then
    seq:add(nn.SpatialSamplingBilinear({owidth=self.imgW,oheight=self.imgH}))
    else
    seq:add(nn.SpatialReSampling({owidth=self.imgW,oheight=self.imgH}))
    end
    self.unfocused_pipeline:add(seq)
   end
end
end
function SpatialPyramid2:focus(x, y, w, h)
   w = w or 1
   h = h or 1
   if x and y then
      self.x = x
      self.y = y
      self.focused = true
      self.winWidth = {}
      self.winHeight = {}
      for i = 1,#self.ratios do
	 self.winWidth[i]  = self.ratios[i] * ((w-1) * self.dW + self.kW)
	 self.winHeight[i] = self.ratios[i] * ((h-1) * self.dH + self.kH)
      end
   else
      self.focused = false
   end
end

function SpatialPyramid2:configureFocus(wImg, hImg)
   for i = 1,#self.ratios do
      local padder = self.focused_pipeline.modules[i].modules[1]
      padder.pad_l = -self.x + math.ceil (self.winWidth[i] /2)
      padder.pad_r =  self.x + math.floor(self.winWidth[i] /2) - wImg
      padder.pad_t = -self.y + math.ceil (self.winHeight[i]/2)
      padder.pad_b =  self.y + math.floor(self.winHeight[i]/2) - hImg
   end
end   

function SpatialPyramid2:checkSize(input)
   for i = 1,#self.ratios do
      if (math.fmod(input:size(2), self.ratios[i]) ~= 0) or
         (math.fmod(input:size(3), self.ratios[i]) ~= 0) then
         error('SpatialPyramid: input sizes must be multiple of ratios')
      end
   end
end
 
function SpatialPyramid2:updateOutput(input)
   if not self.prescaled_input then
      self:checkSize(input)
   end
   if self.focused then
      self:configureFocus(input:size(3), input:size(2))
      self.output = self.focused_pipeline:updateOutput(input)
   else
      self.output = self.unfocused_pipeline:updateOutput(input)
   end
   return self.output
end

function SpatialPyramid2:updateGradInput(input, gradOutput)
   if self.focused then
      self.gradInput = self.focused_pipeline:updateGradInput(input, gradOutput)
   else
      self.gradInput = self.unfocused_pipeline:updateGradInput(input, gradOutput)
   end
   return self.gradInput
end

function SpatialPyramid2:zeroGradParameters()
   self.focused_pipeline:zeroGradParameters()
   self.unfocused_pipeline:zeroGradParameters()
end

function SpatialPyramid2:accGradParameters(input, gradOutput, scale)
   if self.focused then
      self.focused_pipeline:accGradParameters(input, gradOutput, scale)
   else
      self.unfocused_pipeline:accGradParameters(input, gradOutput, scale)
   end
end

function SpatialPyramid2:updateParameters(learningRate)
   if self.focused then
      self.focused_pipeline:updateParameters(learningRate)
   else
      self.unfocused_pipeline:updateParameters(learningRate)
   end
end

function SpatialPyramid2:type(type)
   parent.type(self, type)
   self.focused_pipeline:type(type)
   self.unfocused_pipeline:type(type)
   return self
end

function SpatialPyramid2:parameters()
   if self.focused then
      return self.focused_pipeline:parameters()
   else
      return self.unfocused_pipeline:parameters()
   end
end

function SpatialPyramid2:__tostring__()
   if self.focused then
      local dscr = tostring(self.focused_pipeline):gsub('\n', '\n    |    ')
      return 'SpatialPyramid (focused)\n' .. dscr
   else
      local dscr = tostring(self.unfocused_pipeline):gsub('\n', '\n    |    ')
      return 'SpatialPyramid (unfocused)\n' .. dscr
   end
end


----build the net
net = nn.Sequential()
local Pyramid = nn.SpatialPyramid2(ratios, processors, imgheight, imgwidth, kW, kH, dW, dH, xDimIn, yDimIn, xDimOut, yDimOut) 
net:add(Pyramid)
net:add(nn.JoinTable(1))
net:add(nn.View(-1):setNumInputDims(2))
net:add(nn.Transpose({1,2}))----transform the Feature maps into a matrix, the rows number is the pixel number, and the column number is the number of feature maps. So, each pixel is represented by a vector.
net:add(nn.Linear(Nstates[#Nstates]*(#ratios), nClass)) ---- a linear map that transfer each pixel feature into a nClass x 1 vector. Then We can use LogSoftMax to compute the probility of the pixel for each class.    


--net:add(nn.logSoftMax())
--criterion=nn.ClassNLLCriterion()


