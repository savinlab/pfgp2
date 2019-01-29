%%% Script for testing 3D GP regression on synthetic data

clear all, close all, clc
rng(17435);

% Create synthetic dataset
n1 = 256;
n2 = 256;
n3 = 50;
T = 20000;
fr_true = sample_tuning_fn(n1, n2, n3);

% Sample position values and population spike values
x_pos = sample_position_vals(T);
x_pop = sample_pop_spikes(T);
x = [x_pos, x_pop];

% Sample corresponding spike counts
y = sample_spike_counts(fr_true, x);

% Run GP regression on data
opt.x_min = [0.0, 0.0, 0.0];
opt.x_max = [256.0, 256.0, 50.0];
opt.ng = [32, 32, 26];
opt.ne = [32, 32, 26];
opt.sm_q = 5;
opt.n_hyp_restarts = 1;
[pf, dbg] = pfgp_3d(y, x, opt);

% Plot ground truth vs GP estimate
plot_results_latent(fr_true, x, y, pf, opt.x_max, 1:10);

% Save results
%saveas(gcf, 'demo_3d_plot.png');
%save('demo_3d_results.mat');


function [f] = sample_tuning_fn(n1, n2, n3)
% Generate a (neurally realistic) random tuning function

% Set basic 2D parameters
x1 = (1:1:n1)'; 
x2 = (1:1:n2)';
[X, Y] = meshgrid(x1, x2);
lbda = 80; 
Ng = 6;
xodd = (1:Ng) * lbda - 45;

% generate 2D function
fr = zeros(n1, n2);
sd = rand - 0.5; 
sig = [1, sd; sd, 1] / 160;
for i = 1:Ng
    for j = 1:Ng
        mux = xodd(i);
        muy = xodd(j) +  mod(i, 2) * lbda * cos(pi / 3);
        xx = [X(:) - mux, Y(:) - muy];
        fr = fr + reshape(exp(-sum(xx' .* (sig * xx'))'), n1, n2);
    end
end
f_2d = 0.1 + 10 * fr / max(fr(:));

% Add third dimension
x3 = reshape(1:1:n3, 1, 1, n3); %????
f = 0.05 * f_2d .* x3;

end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));
end


function [x] = sample_pop_spikes(n_pts)
% Sample 'population spike' variable (third domain dimension)

%p_raw = 1 ./ (1:50);
p_raw = ones(1, 50);

p_vals = p_raw / sum(p_raw);

x = sample_discrete(p_vals, n_pts);

end


function [y] = sample_spike_counts(fr_true, x)
% Sample spikes from Poisson distribution

lin_idx = sub2ind(size(fr_true), x(:, 1), x(:, 2), x(:, 3));
fr_vals = fr_true(lin_idx);
y = poissrnd(fr_vals);

end


function plot_fr_true(fr_true, cbar_lims)
% Plot ground truth tuning function

imagesc(fr_true');
axis square;
axis off;
axis xy;
caxis(cbar_lims);
colorbar;

end


function plot_raw_data(x, y, cbar_lims)
% Plot raw position data

n_pts = size(x, 1);
plot(x(1:2:end, 1), x(1:2:end, 2), 'Color', 0.7 * [1, 1, 1]);
axis square;
axis off;
hold on;
idx = and(y > 0, mod(1:n_pts, 2)' == 1);
scatter(x(idx, 1), x(idx, 2), 9, y(idx), 'filled');
caxis(cbar_lims);
colorbar;

end


function plot_mtuning(mtuning, cbar_lims)
% Plot mean of tuning function estimator

imagesc(mtuning');
axis square;
axis off;
axis xy;
caxis(cbar_lims);
colorbar;

end


function plot_sdtuning(sdtuning, cbar_lims)
% Plot stddev of tuning function estimator

imagesc(sdtuning');
axis square;
axis off;
axis xy;
caxis(cbar_lims);
colorbar;

end


function plot_results_tuning(fr_true, x, y, pf, x_max, grid_indices)
% Plot results of demo

grid_max = size(pf.mtuning, 3);
n_grid_indices = size(grid_indices, 2);

figure();

for i = 1:n_grid_indices

    grid_idx = grid_indices(i);

    fr_true_slice = fr_true(:, :, grid_idx);
    slice_idx = (x(:, 3) == grid_idx);
    x_slice = x(slice_idx, 1:2);
    y_slice = y(slice_idx, 1);

    mtuning_slice = pf.mtuning(:, :, grid_idx);
    vartuning_slice = pf.vartuning(:, :, grid_idx);

    cbar_lims_fr = [min(fr_true_slice(:)), max(fr_true_slice(:))];
    cbar_lims_sd = [0, max(sqrt(vartuning_slice(:)))];
    cbar_lims_raw = [0, 5];

    subplot(n_grid_indices, 4, i * 4 - 3);
    plot_fr_true(fr_true_slice, cbar_lims_fr);
    title(sprintf('%d/%d', grid_idx, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 2);
    plot_raw_data(x_slice, y_slice, cbar_lims_raw);
    title(sprintf('%d/%d', grid_idx, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 1);
    plot_mtuning(mtuning_slice, cbar_lims_fr);
    title(sprintf('%d/%d', grid_idx, grid_max));

    subplot(n_grid_indices, 4, i * 4);
    plot_sdtuning(sqrt(vartuning_slice), cbar_lims_sd);
    title(sprintf('%d/%d', grid_idx, grid_max));

end

end


function plot_results_latent(fr_true, x, y, pf, x_max, grid_indices)
% Plot results of demo

grid_max = size(pf.mtuning, 3);
n_grid_indices = size(grid_indices, 2);

figure();

for i = 1:n_grid_indices

    grid_idx = grid_indices(i);

    fr_true_slice = fr_true(:, :, grid_idx);
    slice_idx = (x(:, 3) == grid_idx);
    x_slice = x(slice_idx, 1:2);
    y_slice = y(slice_idx, 1);

    log_fr_true_slice = log(fr_true_slice);
    fmu_slice = pf.fmu(:, :, grid_idx);
    fsd2_slice = pf.fsd2(:, :, grid_idx);

    cbar_lims_fr = [min(log_fr_true_slice(:)), max(log_fr_true_slice(:))];
    cbar_lims_sd = [0, max(sqrt(fsd2_slice(:)))];
    cbar_lims_raw = [0, 5];

    subplot(n_grid_indices, 4, i * 4 - 3);
    plot_fr_true(log_fr_true_slice, cbar_lims_fr);
    title(sprintf('%d/%d', grid_idx, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 2);
    plot_raw_data(x_slice, y_slice, cbar_lims_raw);
    title(sprintf('%d/%d', grid_idx, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 1);
    plot_mtuning(fmu_slice, cbar_lims_fr);
    title(sprintf('%d/%d', grid_idx, grid_max));

    subplot(n_grid_indices, 4, i * 4);
    plot_sdtuning(sqrt(fsd2_slice), cbar_lims_sd);
    title(sprintf('%d/%d', grid_idx, grid_max));

end

end
