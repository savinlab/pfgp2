%%% Script for testing 3D GP regression on simple synthetic data

clear all, close all, clc
rng(17435);

x_min = [1.0, 1.0, 1.0];
x_max = [256.0, 256.0, 4.0];
dims = [256, 256, 4];
n_pts = 7000;

% Define true firing rate function
[X_1, X_2, X_3] = meshgrid( ...
    linspace(x_min(1), x_max(1), dims(1)), ...
    linspace(x_min(2), x_max(2), dims(2)), ...
    linspace(x_min(3), x_max(3), dims(3)) ...
);
fr_true = 0.2 * exp((cos(X_1 ./ 15.0) + cos(X_2 ./ 15.0))) .* X_3;

% Sample data points
n_vals = prod(dims);
x_ind = randi([1, n_vals], [n_pts, 1]);
x_sub = ind2sub(dims, x_ind);
x = [X_1(x_sub), X_2(x_sub), X_3(x_sub)];

% Sample corresponding spike counts
f = fr_true(x_sub);
y = poissrnd(f);

% Run GP regression on data
opt.x_min = x_min;
opt.x_max = x_max;
opt.ng = [32, 32, 4];
opt.ne = [32, 32, 4];
opt.sm_q = 5;

[hyp, hyp_dbg] = mle_hyp_3d(y, x, opt);
[pf, dbg] = pfgp_3d(y, x, opt, hyp);

% Plot ground truth vs GP estimate
plot_results(fr_true, x, y, pf, x_max);


%% Helper functions

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


function plot_results(fr_true, x, y, pf, x_max)
% Plot results of demo

grid_max = size(pf.mtuning, 3);
inc = x_max(3) / grid_max;
grid_indices = 1:4;
n_grid_indices = size(grid_indices, 2);

figure();

for i = 1:n_grid_indices

    grid_idx = grid_indices(i);
    ps_max = grid_idx * inc + 1;
    ps_min = ps_max - inc;

    fr_true_slice = fr_true(:, :, ps_min);
    slice_idx = (x(:, 3) >= ps_min) & (x(:, 3) < ps_max);
    x_slice = x(slice_idx, 1:2);
    y_slice = y(slice_idx, 1);

    mtuning_slice = pf.mtuning(:, :, grid_idx);
    vartuning_slice = pf.vartuning(:, :, grid_idx);

    cbar_lims = [min(fr_true_slice(:)), max(fr_true_slice(:))];

    subplot(n_grid_indices, 4, i * 4 - 3);
    plot_fr_true(fr_true_slice, cbar_lims);
    title(sprintf('%d/%d', ps_min, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 2);
    plot_raw_data(x_slice, y_slice, cbar_lims);
    title(sprintf('%d:%d/%d', ps_min, ps_max, x_max(3)));

    subplot(n_grid_indices, 4, i * 4 - 1);
    plot_mtuning(mtuning_slice, cbar_lims);
    title(sprintf('%d/%d', grid_idx, grid_max));

    subplot(n_grid_indices, 4, i * 4);
    plot_sdtuning(sqrt(vartuning_slice), cbar_lims);
    title(sprintf('%d/%d', grid_idx, grid_max));

end

end
