----input includes:
----nodePot: an n*s matrix, n is the number of nodes, s is the number of states; the potentional values of the node
----edgePot: an s*s*m array, s is the number of states, m is the number of edges; the potentional values of the edges.
----edgeStructure: edgeEnds, E, V, nNodes, nEdges, nStates. a table including structure information.

function makeClampedPotentials(nodePot, edgePot, edgeStruct, clamped)
 local nNodes = nodePot:size(1)
 local maxState = nodePot:size(2)
 local nEdges = edgePot:size(3)
 local edgeEnds = edgestruct.edgeEnds
 local  V = edgeStruct.V
 local E = edgeStruct.E
 local nStates = edgeStruct.nStates
  
  local nodeNum = 1
  local nodeMap = torch.zeros(nNode,1)
  for n = 1, nNodes do
    if clamped(n) == 0 then
      local edges = E[{V[n],V[n+1]-1}]
      for i = 1, edges:size(1) do
        local n1 = edgeEnds[edges[i]][1]
        local n2 = edgeEnds[edges[i]][2]
        
        if n == edgeEnds[edges[i]][1] then
          if clamped[n2] ~= 0 then
            nodePot[{n,{1,nStates[n]}}] =  nodePot[{n,{1,nStates[n]}}] .*edgePot[{{1:nStates[n]},clamped[n2],edges[i]}]:transpose(1,2)
          end
        else
          if clamped[n1] ~= 0 then
            nodePot[{n,{1,nStates[n]}}] = nodePot[{n,{1,nStates[n]}}].*edgePot[{clamped[n1],{1,nStates[n]},edges[i]}]
          end
        end
      end
      nodeMap[n] = nodeNum
      nodeNum = nodeNum + 1
    end
  end
  
  local killedNodes = find(clamped~=0) ----to do: translate in torch 
  local killedEdges = torch.zeros(nEdges,1)
  local k = 0
  local edgeNum = 1
  local edgeMap = torch.zeros(nEdges,1)
  
  for e  = 1,nEdges do 
    local n1 = edgeEnds[e][1]
    local n2 = edgeEnds[e][2]
    if clamped[n1] ~= 0 or clamped[n2] ~=0 then
      k = k+1
      killedEdges[k][1] = e
    else
      edgeEnds[e][1] = nodeMap[n1]
      edgeEnds[e][2] = nodeMap[n2]
      edgeMap[e] = edgeNum
      edgeNum = edgeNum + 1
    end
  end
  
  
    
      
        
            
        
          
            
          
        
        