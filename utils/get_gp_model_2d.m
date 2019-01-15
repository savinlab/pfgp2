function [model] = get_gp_model_2d(opt)
% Return 2D GP model for given 'opt' settings
% 
% Args:
%     opt (struct): Global parameters
%         x_min (float): Minimum position value (default value: 1.0)
%         x_max (float): Maximum position value (default value: 256.0)
%         ng (int): Number of points to use for each dimension of kernel grid 
%             (default value: 256)
%         ne (int): Number of points to use for each dimension of test grid
%             (default value: 64)
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel (default value: false)
%
% Returns:
%     model (struct): GP model parameters
%         sm_q (int): Number of mixture components (for SM kernel)
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
    model.sm_q = 5;
    model.cov = {@apxGrid, {{@covSM, model.sm_q}, {@covSM, model.sm_q}}, kg};
end

end
