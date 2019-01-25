%%% Within-model demo of pfgp_2d function

clear all, close all, clc
rng(17436);

% Create synthetic dataset

n_pts = 7000;
data_dims = [256, 256];

opt.x_min = 1.0;
opt.x_max = 256.0;
opt.use_se = false;
opt.sm_q = 5;
opt.ng = 256;
opt.ne = 64;

hyp = get_hyp_init_2d(opt);
fr_true = sample_tuning_fn(hyp, opt, data_dims, 2);
x = sample_position_vals(n_pts);
y = sample_spike_counts(fr_true, x, data_dims);

% Run GP regression on data
[pf, dbg] = pfgp_2d(y, x, opt);

% Plot ground truth vs GP estimate
plot_results(fr_true, x, y, pf);


function [t] = sample_tuning_fn(hyp, opt, dims, inc)

s_dims = dims / inc;
x_1 = linspace(opt.x_min, opt.x_max, s_dims(1))';
x_2 = linspace(opt.x_min, opt.x_max, s_dims(2))';

[f_s, ~] = sample_hyp_sm_2d(opt.sm_q, hyp, {x_1, x_2}, 1);
f_s_grid = reshape(f_s, s_dims);
f = expand_grid_2d(f_s_grid, inc);

t = exp(f);

end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));
end


function [y] = sample_spike_counts(fr_true, x, dims)
% Sample spikes from Poisson distribution

y = poissrnd(fr_true(sub2ind(dims, x(:,1), x(:,2))));
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


function plot_results(fr_true, x, y, pf)
% Plot results of ground truth recovery experiment

%pf_fn_vals = [pf.mtuning(:); sqrt(pf.vartuning(:))];
%fr_cbar_lims = [min(fr_true(:)), max(fr_true(:))];
%pf_cbar_lims = [min(pf_fn_vals(:)), max(pf_fn_vals(:))];

all_vals = [fr_true(:); pf.mtuning(:); sqrt(pf.vartuning(:))];
cbar_lims = [min(all_vals), max(all_vals)];

figure();
subplot(141);
plot_fr_true(fr_true, cbar_lims);
subplot(142);
plot_raw_data(x, y);
subplot(143);
plot_mtuning(pf.mtuning, cbar_lims);
subplot(144);
plot_sdtuning(sqrt(pf.vartuning), cbar_lims);

end
