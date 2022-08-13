function lgraph = LSTM_AutoEncoder(input_size)


input_layer = sequenceInputLayer(input_size,'Name','seq_input');

lstm_layer_1 = lstmLayer(10,'Name','lstm_layer_1');

fc_layer_1 = fullyConnectedLayer(80,'Name','FC1');

fc_layer_2 = fullyConnectedLayer(50,'Name','FC2');

fc_layer_3 = fullyConnectedLayer(10,'Name','FC3');

fc_layer_4 = fullyConnectedLayer(50,'Name','FC4');

fc_layer_5 = fullyConnectedLayer(80,'Name','FC5');

lstm_layer_2 = lstmLayer(input_size,'Name','lstm_layer_2','OutputMode','sequence');

regression_layer = regressionLayer('Name','Regression_output');

layers = [input_layer,lstm_layer_1,fc_layer_2,fc_layer_3,fc_layer_4,lstm_layer_2,regression_layer];
lgraph = layerGraph(layers);
figure
plot(lgraph)