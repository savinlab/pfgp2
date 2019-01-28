%%% Script for testing 2D GP regression on synthetic data

clear all, close all, clc
rng(17435);

% Create synthetic dataset
n1 = 256;
n2 = 256;
T = 7000;
fr_true = sample_tuning_fn(n1, n2);
x = sample_position_vals(T);
y = sample_spike_counts(fr_true, x, n1, n2);

% Run GP regression on data
opt.ng = 256;
opt.ne = 64;
[pf, dbg] = pfgp_2d(y, x, opt);

% Plot ground truth vs GP estimate
plot_results(fr_true, x, y, pf);

% Save results
saveas(gcf, 'demo_2d_plot.png');
save('demo_2d_results.mat');



function [f] = sample_tuning_fn(n1, n2)
% Generate a (neurally realistic) random tuning function

% Construct covariance grid
x1 = (1:1:n1)'; 
x2 = (1:1:n2)';
[X, Y] = meshgrid(x1, x2);
lbda = 80; 
Ng = 6;
xodd = (1:Ng) * lbda - 45;

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

f = 0.1 + 10 * fr / max(fr(:));
end


function [x] = sample_position_vals(n_pts)
% Use random walk to sample realistic trajectory in 2D space

% Generate samples from random walk
rw_smps = get_rnd_walk_ring(0.05, n_pts, 0.25, 0.99, [0, 0.75]);

% Rescale x to unit square
x = ceil(256 / 2 * (rw_smps + 1));
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
% Plot results of ground truth recovery experiment

pf_fn_vals = [pf.mtuning(:); sqrt(pf.vartuning(:))];
fr_cbar_lims = [min(fr_true(:)), max(fr_true(:))];
pf_cbar_lims = [min(pf_fn_vals(:)), max(pf_fn_vals(:))];

figure();
subplot(141);
plot_fr_true(fr_true, fr_cbar_lims);
subplot(142);
plot_raw_data(x, y);
subplot(143);
plot_mtuning(pf.mtuning, pf_cbar_lims);
subplot(144);
plot_sdtuning(sqrt(pf.vartuning), pf_cbar_lims);

end
