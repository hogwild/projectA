require 'torch'
require 'nn'




----define a nobias Linear regression
local Linear_nobias, parent = torch.class('nn.Linear_nobias', 'nn.Module')

function Linear_nobias:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize):fill(0)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   self:reset()
end

function Linear_nobias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function Linear_nobias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear_nobias:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Linear_nobias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
     local nframe = input:size(1)
     local nunit = self.bias:size(1)
     if nunit == 1 then
       self.gradWeight:select(1,1):addmv(scale,input:t(), gradOutput:select(2,1))
      ----self.gradWeight:addmm(scale, gradOutput:t(), input)
    else
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      ----self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
Linear_nobias.sharedAccUpdateGradParameters = Linear_nobias.accUpdateGradParameters


function Linear_nobias:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end



----test
--[[a=nn.Linear_nobias(10,5)
b=torch.rand(10)
c=a:forward(b)
print(c)
d=a:backward(b,c)
print(d)--]]


----build the 2-layer net
local inSize = net:listModules()[5].output:size(2)
local outSize_1 = 1024--math.ceil(inSize*0.5)
local nCat = 25
Descriptor = nn.Sequential()
Descriptor:add(nn.Linear(inSize, outSize_1))
Descriptor:add(nn.Tanh())
Descriptor:add(nn.Linear_nobias(outSize_1, nCat)) 

----test2
--[[a=torch.rand(25,700)
b=Descriptor:forward(a)
print(b)
c=Descriptor:backward(a,b)
print(c)--]]
