function [model] = get_gp_model_2d(opt)
% Return 2D GP model for given 'opt' settings
% 
% Args:
%     opt (struct): Global parameters
%         x_min (float): Minimum position value 
%         x_max (float): Maximum position value
%         ng (int): Number of points to use for each dimension of kernel grid 
%         ne (int): Number of points to use for each dimension of test grid
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel
%         sm_q (int): Number of mixture components for SM kernel. Only
%             required if SM kernel is being used.
%
% Returns:
%     model (struct): GP model parameters
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)

% Grid for kernel
kg = { ...
    linspace(opt.x_min, opt.x_max, opt.ng)',
    linspace(opt.x_min, opt.x_max, opt.ng)' ...
};

% GP model parameters
model.lik = {@likPoisson, 'exp'};
model.mean = {@meanZero};
if opt.use_se
    model.cov = {@apxGrid, {{@covSEiso}, {@covSEiso}}, kg};
else
    model.cov = {@apxGrid, {{@covSM, opt.sm_q}, {@covSM, opt.sm_q}}, kg};
end

end
