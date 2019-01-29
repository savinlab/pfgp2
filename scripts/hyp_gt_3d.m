%%% Ground-truth hyperparameter recovery with correct init

clear all, close all, clc
rng(17436);

% Create synthetic dataset
data_dims = [256, 256, 12];
opt.x_min = [1.0, 1.0, 1.0];
opt.x_max = [256.0, 256.0, 12.0];
opt.use_se = false;
opt.sm_q = 5;
opt.n_hyp_restarts = 1;
opt.ng = [32, 32, 12];
opt.ne = [32, 32, 12];;

% Sample true hyperparameters
hyp_true = get_hyp_init_3d(opt);

% Test hyperparameter estimation with varying number of data points
n_pts_vals = [5000, 7500, 10000, 12500, 15000];
for i = 1:size(n_pts_vals, 2)

    n_pts = n_pts_vals(i);

    fr_true = sample_tuning_fn(hyp_true, opt, data_dims, 4);
    pos_vals = sample_position_vals(n_pts);
    pop_spikes = sample_pop_spikes(n_pts);
    x = [pos_vals, pop_spikes];
    y = sample_spike_counts(fr_true, x);

    % Run hyperparameter estimation with true values as init
    [hyp_est, dbg] = mle_hyp_3d(y, x, opt, hyp_true);

    % Compute error
    hyp_dist = norm(hyp_true.cov - hyp_est.cov, 2);
    hyp_true_norm = norm(hyp_true.cov, 2);
    hyp_est_norm = norm(hyp_est.cov, 2);
    hyp_err_vals(i) = hyp_dist / (hyp_true_norm * hyp_est_norm);

end

% Plot results
plot(n_pts_vals, hyp_err_vals, '-.*');
xlabel('num. data points');
ylabel('normalized error');


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


