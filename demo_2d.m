%%% Script for testing 2D GP regression on synthetic data

clear all, close all, clc

run('gpml/startup.m');
addpath('utils');
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


function plot_results(fr_true, x, y, pf)
% Plot results of ground truth recovery experiment

figure();
subplot(141);
imagesc(fr_true');
title('ground truth');
axis square; 
axis xy;

subplot(142);
n_pts = size(x, 1);
plot(x(1:2:end, 1), x(1:2:end, 2), 'Color', 0.7 * [1, 1, 1]);
axis square;
axis off;
hold on;
idx = and(y > 0, mod(1:n_pts, 2)' == 1);
scatter(x(idx, 1), x(idx, 2), 9, y(idx), 'filled');
title('raw data ');

subplot(143);
imagesc(pf.mtuning');
title('posterior mean estimate');
% caxis([0 5])
axis square;
axis xy;

subplot(144);
imagesc(sqrt(pf.vartuning)');
title('posterior sd ');
axis square;
axis xy;
end
