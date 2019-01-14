%%% Script for running all unit tests

clear all, close all, clc

run('gpml/startup.m');
addpath('utils');

test_scripts = dir('test/*.m'); 
for i = 1:size(test_scripts, 1)
    
    test_script = test_scripts(i, 1);
    fpath = sprintf('test/%s', test_script.name);
    runtests(fpath);

end
