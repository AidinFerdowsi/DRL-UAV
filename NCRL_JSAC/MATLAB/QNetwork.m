function lgraph = QNetwork(state_per_node,const_size)

input_layer1 = imageInputLayer([state_per_node + state_per_node+const_size,1,1],'Name','Input_Nodes');
% input_layer2 = imageInputLayer([3,1,1],'Name','Input_Constraints');

p = 5;

fc_layer = fullyConnectedLayer(p,'Name','FC');
% concat_layer = concatenationLayer(1,2,'Name','Concat');
relu_layer = reluLayer('Name','Relu');
fc_layer_2 = fullyConnectedLayer(1,'Name','FC2');
% regression_layer = regressionLayer('Name','Regression_output');
softmax_layer = regressionLayer('Name','Softmax_output');
% layers = [input_layer1,fc_layer,concat_layer,relu_layer,regression_layer];
% layers = [input_layer1,fc_layer,relu_layer,fc_layer_2,regression_layer];
layers = [input_layer1,fc_layer,relu_layer,fc_layer_2,softmax_layer];
lgraph = layerGraph(layers);
% lgraph = addLayers(lgraph,input_layer2);
% lgraph = connectLayers(lgraph,'Input_Constraints','Concat/in2');
% figure
% plot(lgraph);





