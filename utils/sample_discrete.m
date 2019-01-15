function [smps] = sample_discrete(p_vals, n_smps)
% Sample values from an arbitrary discrete distribution.
%
% Args:
%     p_vals (1xN float array): Probability values for distribution. Values
%         must add up to 1.
%     n_smps (int): Number of samples to generate
% 
% Returns:
%     smps ((n_smps)x1 int array): Samples from distribution.

rnd_smps = rand(n_smps, 1);
cdf = cumsum(p_vals);
smps = discretize(rnd_smps, [0.0, cdf]);

end
