%%% Script for plotting initial hyperparameter values for pfgp_2d

clear all, close all, clc
rng(17436);

opt.x_min = 1.0;
opt.x_max = 256.0;
opt.use_se = false;
opt.sm_q = 5;
opt.ng = [256, 256];
opt.ne = [64, 64];

n_smps = 15;

figure();
for i = 1:n_smps

    hyp = get_hyp_init_2d(opt);

    ns = [64, 64];
    x_vecs = {
        linspace(opt.x_min, opt.x_max, ns(1))', ...
        linspace(opt.x_min, opt.x_max, ns(2))' ...
    };
    [f_vec, ~] = sample_hyp_sm_2d(opt.sm_q, hyp, x_vecs, 1);
    f = reshape(f_vec, ns);

    subplot(n_smps / 3, 3, i);
    imagesc(f);
    axis square;
    axis off;

end
