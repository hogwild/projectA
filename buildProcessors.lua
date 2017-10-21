require 'nnx'

function buildProcessor(nProcessors,nChannels,Nstates,fanIn,filtSize,poolSize)
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
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nfeats,nstates[1],fanin[1]),filtsize,filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --This MaxPooling can be substituted by a LPPooling + SubtractiveNormalization
    --stage2: covolution layer
    model: add(nn.SpatialConvolutionMap(nn.tables.random(nstates[1],nstates[2],fanin[2]),filtsize,filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize)) --same to stage1
  --stage3: a simple filter bank
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nstates[2],nstates[3],fanin[3]),filtsize,filtsize))
  --- add the model to processors
    processors[i]=nn.Sequential()
    processors[i]:add(model)
    else
    processors[i]=processors[1]:clone('weight','bias','gradWeight','gradBias')
    end
  end
  return processors
end


a=buildProcessor(3,3,{16,64,256},{1,8,32},7,2)
