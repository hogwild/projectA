 require 'nn'
 
 local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(gamma)
   parent.__init(self)
   self.gamma = gamma or 0.5
end

function RBF:updateOutput(input)
  local a = input
  assert(a:nDimension() == 2, 'input tensor must be 2D')
  local N = a:size(1)
  self.output:resize(N,N)
  local tmp = {}
  for i = 1 , N do
    tmp = a[i]:repeatTensor(N,1)-a
    tmp:pow(2)
    self.output[{{},{i}}] = torch.sum(tmp,2)*self.gamma
   end
  self.output:exp()
  return self.output
end

function RBF:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):fill(0)
  assert(gradOutput:nDimension() == 2, 'arguments must be a 2D Tensor')
  assert(gradOutput:size(1)==gradOutput:size(2), 'arguments must be a n*n Tensor')
  local tmp=torch.Tensor() 
  torch.cmul(tmp,self.output,gradOutput)
  tmp:mul(self.gamma)
  --tmp:pow(2-1)
  tmp:cmul(tmp):mul(2)
  local N = tmp:size(1)
  local gradInput=torch.Tensor(input:size(2))
  for i = 1, N do
    for j = 1, N do
      gradInput:copy(input[i])
      gradInput:add(-1,input[j])
      gradInput:mul(tmp[i][j])
      self.gradInput[{{i},{}}]:add(2,gradInput)  
    end
  end
  return self.gradInput
end


mlp=nn.RBF(1)
x = torch.rand(3,2)
y = torch.rand(3,3)
a=mlp:forward(x)
b=mlp:backward(x,y)
