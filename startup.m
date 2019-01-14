%%% Startup script for pfgp2 package

fprintf('Executing pfgp2 startup script...\n')
pkg_root_dir = fileparts(mfilename('fullpath'));
fprintf('Loading pfgp2 package from %s\n', pkg_root_dir);
addpath(pkg_root_dir);
addpath([pkg_root_dir, '/utils']);
fprintf('Done.\n');
