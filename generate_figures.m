clear; close all

addpath(genpath('./PGNLR'))
addpath(genpath('./helper_functions/'))

%% Choose session
result_folder = '013118-Position-Velocity_white';
result_file = 'cherry_bimanualCenterOut_20120316145037_binned_0p1_GNLR_decoding_results_maxiters3000_o1_c5_d3_nc8';
kinematic = 1; % which kinematic to create figures for (LH x, LH y, RH x, RH y; double-check ordering)
do_eigen = 0;

%% Load data and eigenvectors; reconstruct W from W~ (the W acting on PCA'd data); generate predictions

results_file_split = strsplit(result_file, {'_', '.'});

session_date = results_file_split{3}(1:8);
fname = ['F:\Bimanual_GNLR\Results\' result_folder '\' session_date '\' result_file '.mat'];
[data, ~, badidx] = generate_data_for_figures(session_date, result_file, fname, do_eigen);

load(fname)
model = gnlr.models{kinematic};

% these should match what you get from setting do_eigen = 1, which directly
% uses the code that was used to process the data
num_pcs = find(cumsum(eigvals./sum(eigvals)) > 0.8, 1);
data.X1 = bsxfun(@minus, [data.X1_; data.X2_], mean([data.X1_; data.X2_], 1))*eigvecs(:, 1:num_pcs)*sqrt(diag(1./eigvals(1:num_pcs)));
data.X2 = data.X1(size(data.X1_, 1)+1:end, 1:num_pcs);
data.X1 = data.X1(1:size(data.X1_, 1), 1:num_pcs);
Y = data.Y2_(:, kinematic);

model.canoncorrU(data.X2);

reconW = eigvecs(:, 1:model.NR)*diag(sqrt(eigvals(1:num_pcs)))*model.W.mu; % + mean(data.X1_(:, 1));
reconWf = model.getfilter()*diag(1./sqrt(eigvals(1:num_pcs)))*eigvecs(:, 1:model.NR)';
[Ypred, Upred] = model.getPredictions(data.X2);
% reconX = Upred.mu'*reconW';

%% Generate plots

c = str2double(results_file_split{11}(2:end)); % number of lags
D = str2double(results_file_split{12}(2:end)); % dimension of nonlinearity
NC = str2double(results_file_split{13}(3:end)); % number of clusters
% makes it so that reconW has the same number of columns as the original X
% (before the badidxs were removed) by setting columns which had badidx ==
% 1 to zero
if ~isempty(badidx)
    reconWnew = zeros(size(reconW, 1) + length(badidx), size(reconW, 2));
    reconWnew(setdiff(1:size(reconWnew, 1), badidx), :) = reconW;
    reconW = reconWnew;
    clear reconWnew
end
NR_orig = size(reconW, 1); % number of units times number of lags, different from number of regressors after whitening

reconW_ = reshape(reconW, NR_orig/c, c, D);

%%% Make scatterplots of u vs y (both true and predicted) for each u
colors = jet(NC);
[~, clust_label] = max(Upred.p, [], 1) ;
[sp_nr, sp_nc] = BestArrayDims(D);
figure;
for i = 1:D
    subplot(sp_nr, sp_nc, i)
    for k = 1:NC
        scatter(Upred.mu(i, clust_label == k), Ypred.mu(clust_label == k), 50, colors(k, :), '.')
        hold on;
    end
    scatter(Upred.mu(i, :), Y, 20, [0 0 0], '.')
    xlabel(sprintf('u_%d', i))
    ylabel('y')
end
mtit('Scatter plots of Y vs each u')

%%% Make 3D scatter plots for pairs of u's
if D > 1
    figure;
    [sp_nr, sp_nc] = BestArrayDims(D*(D+1)/2 - D);
    count = 1;
    for i = 1:D
        for j = i+1:D
            subplot(sp_nr, sp_nc, count)
            for k = 1:NC
                scatter3(Upred.mu(i, clust_label == k), Upred.mu(j, clust_label == k), Ypred.mu(clust_label == k), 50, colors(k, :), '.')
                hold on;
            end
            scatter3(Upred.mu(i, :), Upred.mu(j, :), Y, 20, [0 0 0], '.')
            xlabel(sprintf('u_%d', i))
            ylabel(sprintf('u_%d', j))
            zlabel('y')
            
            count = count + 1;
        end
    end
    mtit('3D scatter plots of Y vs pairs of u''s')
end

%%% Scatter the u's against each other to identify the domains they
%%% separate
if D > 1
    figure;
    [sp_nr, sp_nc] = BestArrayDims(D*(D+1)/2 - D);
    count = 1;
    for i = 1:D
        for j = i+1:D
            subplot(sp_nr, sp_nc, count)
            for k = 1:NC
                scatter(Upred.mu(i, clust_label == k), Upred.mu(j, clust_label == k), 50, colors(k, :), '.')
                hold on
            end
            xlabel(sprintf('u_%d', i))
            ylabel(sprintf('u_%d', j))
            
            count = count + 1;
        end
    end
    mtit('Scatter plots of pairs of u''s')
end

%%% Plot histogram of columns of reconstructed W, first in one plot, then
%%% in separate subplots.
legend_entries = cell(c, 1);
for i = 1:c
    legend_entries{i} = sprintf('Lag %d', i);
end
for i = 1:size(reconW_, 3)
    figure; hist(reconW_(:, :, i), 20)
    title(sprintf('W_%d', i))
    legend(legend_entries, 'Location', 'Best')
end

% [nr_plot, nc_plot] = BestArrayDims(c);
% for i = 1:D
%     figure
%     for j = 1:c
%         subplot(nr_plot, nc_plot, j)
%         hist(reconW((j-1)*NR_orig/c+1:j*NR_orig/c, i))
% %         xlim([-1 3])
%         title(sprintf('W_%d, lag %d', i, j))
%     end
% end

%%% Make "spaghetti plots", which show the contribution of each neuron as a
%%% function of lag.
[nr_plot, nc_plot] = BestArrayDims(D);
figure;
for i = 1:D
    stds = std(reconW_(:, :, i), [], 2);
    thresh = quantile(stds, 0.95);
    subplot(nr_plot, nc_plot, i); plot(reconW_(stds > thresh, :, i)')
    xlabel('Lag')
    ylabel(sprintf('W_{%d,i}', i))
end
mtit('Modulation of neurons about baseline as a function of lag')

% figure;
% for i = 1:D
%     subplot(nr_plot, nc_plot, i); imagesc(reconW_(:, :, i))
%     xlabel('Lag')
%     ylabel('Unit')
%     colorbar
% end

%%% Assess relative contributions of neurons to different W's
if D > 1
    [nr_plot, nc_plot] = BestArrayDims(D*(D-1)/2);
    count = 1;
    figure
    for i = 1:D
        for j = i+1:D
            subplot(nr_plot, nc_plot, count)
            scatter(std(reconW_(:, :, i), [], 2), std(reconW_(:, :, j), [], 2))
            refline(1, 0)
            xlabel(sprintf('W_%d', i))
            ylabel(sprintf('W_%d', j))
            
            count = count + 1;
        end
    end
    mtit('STDs of neurons across lags for pairs of model dimensions')
end

if D > 1
    count = 1;
    figure;
    colors = jet(c);
    for i = 1:D
        for j = i+1:D
            subplot(nr_plot, nc_plot, count)
            for k = 1:c
                h = scatter(reconW_(:, k, i), reconW_(:, k, j), [], colors(k, :));
%                 set(h, 'MarkerFaceColor', 'flat', 'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', (c-k)/c);
                hold on;
            end
            
            refline(1, 0)
            xlabel(sprintf('W_%d', i))
            ylabel(sprintf('W_%d', j))
            
            count = count + 1;
        end
    end
    legend(legend_entries, 'Location', 'Best')
    mtit('Neuron activation for pairs of model dimensions; color = lag')
end

%%% Report average and standard deviation of neural response
for i = 1:D
    mean_reconW = mean(reconW(:, i));
    std_reconW = std(reconW(:, i));
    fprintf('Mean firing rate of neurons for W_%d: %3.4f\n', i, mean_reconW)
    fprintf('STD of firing rate of neurons for W_%d: %3.4f\n', i, std_reconW)
    
    if mean_reconW < 0
        fprintf('p-value for difference from 0 firing rate for W_%d: %3.4e\n\n', i, tcdf(mean_reconW./std_reconW*sqrt(NR_orig), NR_orig-1))
    else
        fprintf('p-value for difference from 0 firing rate for W_%d: %3.4e\n\n', i, tcdf(mean_reconW./std_reconW*sqrt(NR_orig), NR_orig-1, 'upper'))
    end
end

var_exp = cov([Ypred.mu', Upred.mu']);
var_exp = var_exp(:, 1)

% figure; plot(data.X1_(:, 1), reconX(:, 1), '.')
% xlabel('true data')
% ylabel('reconstructed data')
% title('First neuron, recon vs. truth')

%% Print the figures to PDF files
% figDir = ['..\Results\020918\GNLR parameter summary\Figures\' session_date];
% if ~exist(figDir, 'dir')
%     mkdir(figDir)
% end
% 
% num_figs =  length(findobj('type','figure'));
% for i = 1:num_figs
%     figure(i)
%     fig = gcf;
%     fig.PaperPositionMode = 'auto'
%     fig_pos = fig.PaperPosition;
%     fig.PaperSize = [fig_pos(3) fig_pos(4)];
%     
%     print([figDir '\fig' num2str(i)], '-dpdf')
% end