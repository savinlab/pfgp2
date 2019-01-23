function [hyp_0] = get_hyp_init_3d(opt)
% Compute initial hyperparameter values for 3D GP regression.
%
% Args:
%     opt (struct): Global GP regression parameters. Contains fields:
%         x_min (1x3 float array): Minimum position value
%         x_max (1x3 float array): Maximum position value
%         use_se (bool): Whether to use SE kernel instead of SM kernel
%         sm_q (int, optional): Number of mixture components for SM kernel,
%             if SM kernel if being used.
%
% Returns:
%     hyp_0 (struct): Initial hyperparameter struct (see GPML docs)
%
% Throws:
%     'missing_params' (error): If any required parameters aren't specified.

assert( ...
    all(isfield(opt, {'x_min', 'x_max', 'use_se'})), ...
    'get_hyp_init_3d:missing_params', ...
    'Required opt parameters missing' ...
);

if opt.use_se

    scl = 1 ./ (opt.x_max - opt.x_min);

    % Initial hyperparameter values for SE kernel
    hyp_0.cov = [log([scl(1); 1]); log([scl(2); 1]), log([scl(3); 1])];
    hyp_0.mean = [];
    hyp_0.lik = [];

else

    assert( ...
        isfield(opt, 'sm_q'), ...
        'get_hyp_init_3d:missing_params', ...
        'sm_q parameter is required for SM kernel' ...
    );

    scl = 1 ./ (opt.x_max - opt.x_min);
    w = ones(opt.sm_q, 1) / opt.sm_q;
    m1 = scl(1) * rand(1, opt.sm_q);
    m2 = scl(2) * rand(1, opt.sm_q);
    m3 = scl(3) * rand(1, opt.sm_q);
    v1 = (scl(1) ^ 2) * rand(1, opt.sm_q);
    v2 = (scl(2) ^ 2) * rand(1, opt.sm_q);
    v3 = (scl(3) ^ 2) * rand(1, opt.sm_q);

    % Initial hyperparameter values for SM kernel
    hyp_0.cov = [ ...
        log([w; m1(:); v1(:)]); ...
        log([w; m2(:); v2(:)]); ...
        log([w; m3(:); v3(:)]) ...
    ];
    hyp_0.mean = [];
    hyp_0.lik = [];

end

end
