function [f, dbg] = sample_hyp_sm_2d(sm_q, hyp, x_vecs, n_smps)
% Sample from GP with given hyperparameters

assert(isempty(hyp.mean), 'Nonempty mean value not supported!');

n_hyp = 3 * sm_q;
hyp_1 = hyp.cov(1:n_hyp);
hyp_2 = hyp.cov(n_hyp+1:end);

cov_1 = covSM(sm_q, hyp_1, x_vecs{1});
cov_2 = covSM(sm_q, hyp_2, x_vecs{2});
dbg.cov_1 = cov_1;
dbg.cov_2 = cov_2;

correction = 1e-8;
cov_1_psd = cov_1 + correction .* eye(size(x_vecs{1}, 1));
cov_2_psd = cov_2 + correction .* eye(size(x_vecs{2}, 1));
dbg.cov_1_psd = cov_1_psd;
dbg.cov_2_psd = cov_2_psd;

ch_1 = chol(cov_1_psd, 'lower');
ch_2 = chol(cov_2_psd, 'lower');
ch_cov = kron(ch_2, ch_1);
dbg.ch_cov = ch_cov;

x = apxGrid('expand', x_vecs);
f = ch_cov * randn([size(x, 1), n_smps]);

end
