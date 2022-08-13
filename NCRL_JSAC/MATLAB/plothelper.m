%% Plot vs num_nodes - small number
lb = 0.078 * ones(10,1);
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\disc_vs_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\NCRL_vs_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\lstm_vs_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\random_vs_num_nodes.mat')
plot(1:10,aoi,'color','g','marker','p','linewidth',2);
hold on;
plot(1:10,aoi_lstm,'color','b','marker','x','linewidth',2);
plot(1:10,aoi_disc,'color','m','marker','d','linewidth',2);
plot(1:10,aoi_random,'color','r','marker','s','linewidth',2);
plot(1:10,lb,'color','k','linestyle','--','linewidth',2);
grid on;
xlabel('Number of nodes ($M$)', 'Interpreter' , 'latex', 'FontSize',13);
ylabel('NWAoI', 'Interpreter' , 'latex', 'FontSize',13);
legend({'Proposed NCRL','Proposed LSTM-autoencoder','Discretized DQN','Weight-based Policy','Lowerbound NWAoI'},'Interpreter' , 'latex', 'FontSize',13,'location','best');
removewhitespace;
%% Plot vs num_nodes - large number
lb = 0.078 * ones(9,1);
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\disc_vs_large_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\NCRL_vs_large_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\lstm_vs_large_num_nodes.mat')
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\random_vs_large_num_nodes.mat')
plot(10:5:50,aoi,'color','g','marker','p','linewidth',2);
hold on;
plot(10:5:50,aoi_lstm,'color','b','marker','x','linewidth',2);
plot(10:5:50,aoi_disc,'color','m','marker','d','linewidth',2);
plot(10:5:50,aoi_random,'color','r','marker','s','linewidth',2);
% plot(10:5:50,lb,'color','k','linestyle','--','linewidth',2);
grid on;
xlabel('Number of nodes ($M$)', 'Interpreter' , 'latex', 'FontSize',13);
ylabel('NWAoI', 'Interpreter' , 'latex', 'FontSize',13);
legend({'Proposed NCRL','Proposed LSTM-autoencoder','Discretized DQN','Weight-based Policy'},'Interpreter' , 'latex', 'FontSize',13,'location','best');
removewhitespace;
%% vs Time Constraint
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\vsTimeConstraint.mat')
plot(5:15,aoi_NCRL,'color','g','marker','p','linewidth',2);
hold on;
plot(5:15,aoi_LSTM,'color','b','marker','x','linewidth',2);
plot(5:15,aoi_DRL,'color','m','marker','d','linewidth',2);
plot(5:15,aoi_random,'color','r','marker','s','linewidth',2);
% plot(10:5:50,lb,'color','k','linestyle','--','linewidth',2);
grid on;
xlabel('Time constraint $\tau$ (minutes)', 'Interpreter' , 'latex', 'FontSize',13);
ylabel('NWAoI', 'Interpreter' , 'latex', 'FontSize',13);
legend({'Proposed NCRL','Proposed LSTM-autoencoder','Discretized DQN','Weight-based Policy'},'Interpreter' , 'latex', 'FontSize',13,'location','best');
removewhitespace;
%% vs Speed
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\vsSpeed.mat')
plot(2:2:20,aoi_NCRL,'color','g','marker','p','linewidth',2);
hold on;
plot(2:2:20,aoi_LSTM,'color','b','marker','x','linewidth',2);
plot(2:2:20,aoi_DRL,'color','m','marker','d','linewidth',2);
plot(2:2:20,aoi_random,'color','r','marker','s','linewidth',2);
% plot(10:5:50,lb,'color','k','linestyle','--','linewidth',2);
grid on;
xlabel('UAV speed (m/s)', 'Interpreter' , 'latex', 'FontSize',13);
ylabel('NWAoI', 'Interpreter' , 'latex', 'FontSize',13);
legend({'Proposed NCRL','Proposed LSTM-autoencoder','Discretized DQN','Weight-based Policy'},'Interpreter' , 'latex', 'FontSize',13,'location','best');
removewhitespace;
%% vs Energy
load('C:\Users\walids\Desktop\Federated_UAV\CombinatorialDRL\tf_agent\results\final_results\vsEnergy.mat')
plot(0.1:0.1:1,aoi_NCRL,'color','g','marker','p','linewidth',2);
hold on;
plot(0.1:0.1:1,aoi_LSTM,'color','b','marker','x','linewidth',2);
plot(0.1:0.1:1,aoi_DRL,'color','m','marker','d','linewidth',2);
plot(0.1:0.1:1,aoi_random,'color','r','marker','s','linewidth',2);
plot(0.1:0.1:1,lb,'color','k','linestyle','--','linewidth',2);
grid on;
xlabel('Average node energy (joules)', 'Interpreter' , 'latex', 'FontSize',13);
ylabel('NWAoI', 'Interpreter' , 'latex', 'FontSize',13);
legend({'Proposed NCRL','Proposed LSTM-autoencoder','Discretized DQN','Weight-based Policy','Lowerbound NWAoI'},'Interpreter' , 'latex', 'FontSize',13,'location','best');
removewhitespace;