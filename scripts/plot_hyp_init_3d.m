%%% Script for plotting initial hyperparameter values for pfgp_3d

clear all, close all, clc
rng(17435);

opt.x_min = [1.0, 1.0, 1.0];
opt.x_max = [256.0, 256.0, 10.0];
opt.use_se = false;
opt.sm_q = 5;
opt.ng = [32, 32, 10];
opt.ne = [32, 32, 10];

n_smps = 5;

for i = 1:n_smps

    hyp = get_hyp_init_3d(opt);

    ns = [32, 32, 10];
    x_vecs = {
        linspace(opt.x_min(1), opt.x_max(1), ns(1))', ...
        linspace(opt.x_min(2), opt.x_max(2), ns(2))', ...
        linspace(opt.x_min(3), opt.x_max(3), ns(3))' ...
    };
    [f_vec, ~] = sample_hyp_sm_3d(opt.sm_q, hyp, x_vecs, 1);
    f = reshape(f_vec, ns);

    figure();
    for j = 1:ns(3)
        subplot(ns(3) / 2, 2, j);
        imagesc(f(:, :, j));
        axis square;
        title(sprintf('slice=%d', j));
    end
    suptitle(sprintf('sample=%d', i));

end
