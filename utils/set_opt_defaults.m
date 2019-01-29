function [opt_full] = set_opt_defaults(opt, opt_default)
% Set unset options to default values

all_fields = fieldnames(opt_default);
for i = 1:numel(all_fields)  
    fname = all_fields{i};
    if isfield(opt, fname) 
        opt_full.(fname) = opt.(fname);
    else
        opt_full.(fname) = opt_default.(fname);
    end
end

end
