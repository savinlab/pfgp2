function [f, dbg] = sample_hyp_sm_3d(sm_q, hyp, x_vecs, n_smps)
% Sample from GP with given hyperparameters (3D)

assert(isempty(hyp.mean), 'Nonempty mean value not supported!');

n_hyp = 3 * sm_q;
hyp_1 = hyp.cov(1:n_hyp);
hyp_2 = hyp.cov(n_hyp+1:2*n_hyp);
hyp_3 = hyp.cov(2*n_hyp+1:end);

cov_1 = covSM(sm_q, hyp_1, x_vecs{1});
cov_2 = covSM(sm_q, hyp_2, x_vecs{2});
cov_3 = covSM(sm_q, hyp_3, x_vecs{3});
dbg.cov_1 = cov_1;
dbg.cov_2 = cov_2;
dbg.cov_3 = cov_3;

correction = 1e-8;
cov_1_psd = cov_1 + correction .* eye(size(x_vecs{1}, 1));
cov_2_psd = cov_2 + correction .* eye(size(x_vecs{2}, 1));
cov_3_psd = cov_3 + correction .* eye(size(x_vecs{3}, 1));
dbg.cov_1_psd = cov_1_psd;
dbg.cov_2_psd = cov_2_psd;
dbg.cov_3_psd = cov_3_psd;

ch_1 = chol(cov_1_psd, 'lower');
ch_2 = chol(cov_2_psd, 'lower');
ch_3 = chol(cov_3_psd, 'lower');
ch_cov = kron(ch_3, kron(ch_2, ch_1));
dbg.ch_cov = ch_cov;

x = apxGrid('expand', x_vecs);
f = ch_cov * randn([size(x, 1), n_smps]);

end
