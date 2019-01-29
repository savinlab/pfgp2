%%% Within-model demo of pfgp_3d function

clear all, close all, clc
rng(17436);

% Create synthetic dataset
n_pts = 7000;
data_dims = [256, 256, 12];
opt.x_min = [1.0, 1.0, 1.0];
opt.x_max = [256.0, 256.0, 12.0];
opt.use_se = false;
opt.sm_q = 5;
opt.n_hyp_restarts = 1;
opt.ng = [32, 32, 12];
opt.ne = [32, 32, 12];;

hyp = get_hyp_init_3d(opt);
fr_true = sample_tuning_fn(hyp, opt, data_dims, 4);
x(:, 1:2) = sample_position_vals(n_pts);
x(:, 3) = sample_pop_spikes(n_pts);
y = sample_spike_counts(fr_true, x);

% Run GP regression on data
[pf, dbg] = pfgp_3d(y, x, opt);

% Plot ground truth vs GP estimate
grid_indices = 3:3:12;
plot_results_tuning(fr_true, x, y, pf, opt.x_max, grid_indices);
%saveas(gcf, 'demo_3d_hyp_tuning_plot.png');
plot_results_latent(fr_true, x, y, pf, opt.x_max, grid_indices);
%saveas(gcf, 'demo_3d_hyp_latent_plot.png');

% Save results
%save('demo_3d_hyp_results.mat');


function [t] = sample_tuning_fn(hyp, opt, dims, inc)

s_dims = dims / inc;
x_1 = linspace(opt.x_min(1), opt.x_max(1), s_dims(1))';
x_2 = linspace(opt.x_min(2), opt.x_max(2), s_dims(2))';
x_3 = linspace(opt.x_min(3), opt.x_max(3), s_dims(3))';

[f_s, ~] = sample_hyp_sm_3d(opt.sm_q, hyp, {x_1, x_2, x_3}, 1);
f_s_grid = reshape(f_s, s_dims);
f = expand_grid_3d(f_s_grid, inc);

t = exp(f);

end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);
% Make hole bigger to check that variance works
%rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.40, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));

end


function [x] = sample_pop_spikes(n_pts)
% Sample 'population spike' variable (third domain dimension)

p_raw = ones(1, 12);
p_vals = p_raw / sum(p_raw);

x = sample_discrete(p_vals, n_pts);

end


function [y] = sample_spike_counts(fr_true, x)
% Sample spikes from Poisson distribution

lin_idx = sub2ind(size(fr_true), x(:, 1), x(:, 2), x(:, 3));
fr_vals = fr_true(lin_idx);
y = poissrnd(fr_vals);

end


function plot_fr_true(fr_true, cbar_limits)
% Plot ground truth tuning function

imagesc(fr_true');
title('ground truth');
axis square; 
axis xy;
caxis(cbar_limits);
colorbar;

end


function plot_raw_data(x, y)
% Plot raw position data

n_pts = size(x, 1);
plot(x(1:2:end, 1), x(1:2:end, 2), 'Color', 0.7 * [1, 1, 1]);
axis square;
axis xy;

xlim([min(x(:, 1)), max(x(:, 1))]);
ylim([min(x(:, 2)), max(x(:, 2))]);
hold on;
idx = and(y > 0, mod(1:n_pts, 2)' == 1);
scatter(x(idx, 1), x(idx, 2), 9, y(idx), 'filled');
colorbar;
title('raw data');

end


function plot_mtuning(mtuning, cbar_limits)
% Plot mean of tuning function estimator

imagesc(mtuning');
title('posterior mean estimate');
axis square;
axis xy;
caxis(cbar_limits);
colorbar;

end


function plot_sdtuning(sdtuning, cbar_limits)
% Plot stddev of tuning function estimator

imagesc(sdtuning');
title('posterior sd');
axis square;
axis xy;
caxis(cbar_limits);
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
