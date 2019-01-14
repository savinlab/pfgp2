%%% Script for testing 3D GP regression on synthetic data

clear all, close all, clc

run('gpml/startup.m');
addpath('utils');
rng(17435);

% Create synthetic dataset
n1 = 256;
n2 = 256;
n3 = 50;
T = 7000;
fr_true = sample_tuning_fn(n1, n2, n3);

% Sample position values and population spike values
x_pos = sample_position_vals(T);
x_pop = sample_pop_spikes(T);
x = [x_pos, x_pop];

% Sample corresponding spike counts
y = sample_spike_counts(fr_true, x, n1, n2);

% Run GP regression on data
opt.x_min = [0.0, 0.0, 0.0];
opt.x_max = [256.0, 256.0, 50.0];
opt.ng = [32, 32, 10];
opt.ne = [32, 32, 10];
[pf, dbg] = pfgp_3d(y, x, opt);

% Plot ground truth vs GP estimate
plot_results(fr_true, x, y, pf);


function [f] = sample_tuning_fn(n1, n2, n3)
% Generate a (neurally realistic) random tuning function

% Set basic 2D parameters
x1 = (1:1:n1)'; 
x2 = (1:1:n2)';
[X, Y] = meshgrid(x1, x2);
lbda = 80; 
Ng = 6;
xodd = (1:Ng) * lbda - 45;

% Sample 2D function
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
x3 = reshape(1:1:n3, 1, 1, n3);
f = 0.2 * f_2d .* x3;

end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));
end


function [smps] = sample_discrete(p_vals, n_smps)
% Sample values from a discrete distribution

rnd_smps = rand(n_smps, 1);
cdf = cumsum(p_vals);
smps = discretize(rnd_smps, [0.0, cdf]);

end


function [x] = sample_pop_spikes(n_pts)
% Sample 'population spike' variable (third domain dimension)

p_raw = 1 ./ (1:50);
p_vals = p_raw / sum(p_raw);

x = sample_discrete(p_vals, n_pts);

end


function [y] = sample_spike_counts(fr_true, x, n1, n2)
% Sample spikes from Poisson distribution

y = poissrnd(fr_true(sub2ind([n1, n2], x(:,1), x(:,2))));
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
axis off;
hold on;
idx = and(y > 0, mod(1:n_pts, 2)' == 1);
scatter(x(idx, 1), x(idx, 2), 9, y(idx), 'filled');
colorbar;
title('raw data ');

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


function plot_results(fr_true, x, y, pf)
% Plot results of demo

pop_spike_max = 50;

grid_max = size(pf.mtuning, 3);
inc = pop_spike_max / grid_max;
grid_indices = [1, 2, 3, 4, 5];
n_grid_indices = size(grid_indices, 2);

pf_fn_vals = [pf.mtuning(:); sqrt(pf.vartuning(:))];
fr_cbar_lims = [min(fr_true(:)), max(fr_true(:))];
pf_cbar_lims = [min(pf_fn_vals(:)), max(pf_fn_vals(:))];
figure();

for i = 1:n_grid_indices

    grid_idx = grid_indices(i);
    ps_max = grid_idx * inc + 1;
    ps_min = ps_max - inc;

    fr_true_slice = fr_true(:, :, ps_min);

    slice_idx = (x(:, 3) >= ps_min) & (x(:, 3) < ps_max);
    x_slice = x(slice_idx, 1:2);
    y_slice = y(slice_idx, 1);
    size(x_slice)

    mtuning_slice = pf.mtuning(:, :, grid_idx);
    vartuning_slice = pf.vartuning(:, :, grid_idx);

    subplot(n_grid_indices, 4, i * 4 - 3);
    plot_fr_true(fr_true_slice, fr_cbar_lims);
    subplot(n_grid_indices, 4, i * 4 - 2);
    plot_raw_data(x_slice, y_slice);
    subplot(n_grid_indices, 4, i * 4 - 1);
    plot_mtuning(mtuning_slice, pf_cbar_lims);
    subplot(n_grid_indices, 4, i * 4);
    plot_sdtuning(sqrt(vartuning_slice), pf_cbar_lims);

end

end
