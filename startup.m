%%% Startup script for pfgp2 library

fprintf('executing pfgp2 startup script...\n')
pkg_root_dir = fileparts(mfilename('fullpath'));

fprintf('(gpml library is included in this package)\n');
run(sprintf('%s/gpml/startup.m', pkg_root_dir));

fprintf('loading pfgp2 library from %s...\n', pkg_root_dir);
addpath(pkg_root_dir);
addpath([pkg_root_dir, '/utils']);
addpath([pkg_root_dir, '/scripts']);
