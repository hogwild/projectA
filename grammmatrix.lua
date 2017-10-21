function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  --RBF:
   --[[ local split = nn.SplitTable(1)
  net:add(split)
  local concat = nn.ConcatTable()
  for i = 1, n do
    for j = 1, n do
      t = nn.ConcatTable() 
      t:add(nn.SelectTable(i))
      t:add(nn.SelectTable(j))
      concat:add(t)
    end
  end
  net:add(concat)
  local parallel = nn.ParallelTable()
  for i = 1, n do
    for j = 1, n do
      parallel:add(nn.PairwiseDistance(2))
      end
  end
    net:add(parallel)
    net:add(nn.JoinTable(1))
    net:add(nn.Reshape(n,n))
    net:add(nn.Power(2))
    net:add(nn.MulConstant(g))
    net:add(nn.Exp())
  return net
end
