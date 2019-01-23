function [model] = get_gp_model_3d(opt)
% Return 2D GP model for given 'opt' settings
% 
% Args:
%     opt (struct): Global parameters
%         x_min (1x3 float array): Minimum position values 
%             (default value: [1.0, 1.0, 1.0])
%         x_max (1x3 float array): Maximum position values
%             (default value: [256.0, 256.0, 256.0])
%         ng (1x3 int array): Number of points to use for each dimension of
%             kernel grid (default value: [32, 32, 10])
%         ne (1x3 int array): Number of points to use for each dimension of
%             test grid (default value: [32, 32, 10])
%         use_se (bool): Whether to use squared exponential (SE) kernel
%             instead of spectral mixture (SM) kernel (default value: false)
%         sm_q (int): Number of mixture components (for SM kernel). Only
%             required if SM kernel is being used.
%
% Returns:
%     model (struct): GP model parameters
%         cov (GPML cov): Covariance object (see GPML docs)
%         mean (GPML mean): Mean object (see GPML docs)
%         lik (GPML lik): Likelihood object (see GPML docs)

% Grid for kernel
kg = { ...
    linspace(opt.x_min(1), opt.x_max(1), opt.ng(1))', ...
    linspace(opt.x_min(2), opt.x_max(2), opt.ng(2))', ...
    linspace(opt.x_min(3), opt.x_max(3), opt.ng(3))' ...
};

% GP model parameters
model.lik = {@likPoisson, 'exp'};
model.mean = {@meanZero};
if opt.use_se
    model.cov = {@apxGrid, {{@covSEiso}, {@covSEiso}, {@covSEiso}}, kg};
else
    model.cov = { ...
        @apxGrid, ...
        {{@covSM, opt.sm_q}, {@covSM, opt.sm_q}, {@covSM, opt.sm_q}}, ...
        kg ...
    };
end

end
