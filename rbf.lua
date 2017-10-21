local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(inputSize, outputSize)
   parent.__init(self)
   self.weight = torch.rand(outputSize, inputSize)*2-1
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.input_size=inputSize
   self.output_size=outputSize
end


function RBF:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.output_size):zero()
      for i=1,self.output_size do
         local tmp=input-self.weight:select(1,i)
         self.output[i]=tmp*tmp
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit =self.output_size

      self.output:resize(nframe, nunit):zero()
      for k=1,nframe do
         for i=1,nunit do
            local tmp=input[k]-self.weight:select(1,i)
            self.output[k][i]=tmp*tmp
         end
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function RBF:updateGradInput(input, gradOutput)
   if self.gradInput then
      local tmp=torch.Tensor(self.input_size, self.output_size)
      if input:dim() == 1 then
         self.gradInput:resizeAs(input)
         tmp:copy(self.weight:t()*-2)
         for i=1,self.output_size do
            tmp:select(2,i):add(2,input)
         end
         self.gradInput:addmv(0, 1, tmp, gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:resizeAs(input)
         local nframe = input:size(1)
         for k=1,nframe do
            tmp:copy(self.weight:t()*-2)
            for i=1,self.output_size do
              tmp:select(2,i):add(2,input[i])
            end
            self.gradInput[k]:addmv(0, 1,tmp,gradOutput[k])
         end
      end
      return self.gradInput
   end
end

function RBF:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addmm(scale,torch.diag(gradOutput),torch.addr(2,self.weight,-2,torch.ones(self.output_size),input))
      --print(torch.norm(self.gradWeight))
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      for k=1,nframe do
         self.gradWeight:addmm(scale,torch.diag(gradOutput[k]),torch.addr(2,self.weight,-2,torch.ones(self.output_size),input[k]))
      end
   end

end
