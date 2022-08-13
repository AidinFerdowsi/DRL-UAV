function lgraph = QNetwork_LSTM(num_nodes)

% input_layer1 = imageInputLayer([3 * num_nodes + 4,1,1],'Name','Input_Nodes');
input_layer1 = sequenceInputLayer(num_nodes*2 + 1 + num_nodes + 2 + num_nodes + 4,'Name','Sequential');
% input_layer2 = imageInputLayer([3,1,1],'Name','Input_Constraints');

p = 10;
hl = 100;
lstm_layer = lstmLayer(hl,'Name','lstm_layer','OutputMode','last');

fc_layer = fullyConnectedLayer(p,'Name','FC');
% concat_layer = concatenationLayer(1,2,'Name','Concat');
relu_layer = reluLayer('Name','Relu');
fc_layer_2 = fullyConnectedLayer(num_nodes,'Name','FC2');
regression_layer = regressionLayer('Name','Regression_output');
softmax_layer = softmaxLayer('Name','Softmax_output');
% layers = [input_layer1,fc_layer,concat_layer,relu_layer,regression_layer];
% layers = [input_layer1,fc_layer,relu_layer,fc_layer_2,regression_layer];
layers = [input_layer1,lstm_layer,fc_layer,relu_layer,fc_layer_2,regression_layer];
lgraph = layerGraph(layers);
% lgraph = addLayers(lgraph,[input_layer2,lstm_layer]);
% lgraph = connectLayers(lgraph,'lstm_layer','Concat/in2');
figure
plot(lgraph);





