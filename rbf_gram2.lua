require 'nn'

RBF2, parent = torch.class('nn.RBF2', 'nn.Module')

function RBF2:__init(gamma)
   parent.__init(self)
   self.gamma = gamma or -0.5
end

function RBF2:updateOutput(input)
  local a = input
  assert(a:nDimension() == 2, 'input tensor must be 2D')
  local N = a:size(1)
  self.output:resize(N,N)
  local tmp
  for i = 1 , N do
    tmp = a[i]:repeatTensor(N,1)-a
    tmp:pow(2)
    self.output[{{},{i}}] = torch.sum(tmp,2)*self.gamma
   end
  self.output:exp()
  return self.output
end

function RBF2:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  assert(gradOutput:nDimension() == 2, 'arguments must be a 2D Tensor')
  assert(gradOutput:size(1)==gradOutput:size(2), 'arguments must be a n*n Tensor')
 
local tmp,N, gradInput
  --self.tmp_gradInput:resizeAs(input[1])
  self.gradInput:resizeAs(input)
  tmp=torch.cmul(self.output,gradOutput)
  tmp:mul(self.gamma)
  --tmp:pow(2-1)
  tmp:cmul(tmp):mul(2)
  N = tmp:size(1)
  for i = 1, N do
    gradInput=input[i]:repeatTensor(N,1)-input
    self.gradInput[{{i},{}}] = tmp[{{i},{}}]*gradInput *2 
  end
  return self.gradInput
end
