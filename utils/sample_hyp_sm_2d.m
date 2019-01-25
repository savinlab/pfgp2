function [f, dbg] = sample_hyp_sm_2d(sm_q, hyp, x_vecs, n_smps)
% Sample from GP with given hyperparameters

assert(isempty(hyp.mean), 'Nonempty mean value not supported!');

n_hyp = 3 * sm_q;
hyp_1 = hyp.cov(1:n_hyp);
hyp_2 = hyp.cov(n_hyp+1:end);

cov_1 = covSM(sm_q, hyp_1, x_vecs{1});
cov_2 = covSM(sm_q, hyp_2, x_vecs{2});
cov_mtx = kron(cov_2, cov_1);
dbg.cov_mtx = cov_mtx;

x = apxGrid('expand', x_vecs);
dbg.x = x;

% Sample from MVN distribution
mean_mtx = zeros(n_smps, size(x, 1));
f = (mvnrnd(mean_mtx, cov_mtx))';

end
