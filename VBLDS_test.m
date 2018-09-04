%% overview

% script for synthetic LDS data generation
% and fitting VBLDS

% model:
% x_t+1 ~ N(x_t+1 | Ax_t, Q)
% y_t ~ N(y_t | Cx_T, R)

%% data synthesis 
rng(2018)

mu0 = [1; 1];
sigma0 = .01 * eye(2);

% A = [.7 .02;
%     .05 .6];
A = .8 * eye(2);

% C = eye(2);
% C = [.7 .4;
%     1 .9];
C = [1 0;
    0 1;
    .5 .5];

Q = .1 * eye(size(A,1));
R = .1 * eye(size(C,1));

T = 10; % number trials
t = 100 * ones(T,1); % trial lengths

% data stored in X{trial} = [dim, time] format

[X, Y] = generate(T, t, mu0, sigma0, A, C, Q, R);

Xtrue = [X{:}]';

%% model fitting
rng(2018)
    
k = 2;
d = size(Y{1},1);
T = size(Y,1);
model = VBLDS(k, d);    

iters = 100;
% Y2 = zScore(Y);
model.fit(Y,iters)



%% model diagnostics

model.plotLs % L can decrease
model.negdLs

model.r(Xtrue)
model.R2

model.A.mean 
model.Ctrue
inv(model.A.invU.mean)
inv(model.C.invU.mean) 



%% functions

% synthesis function
function [X, Y] = generate(T, t, mu0, sigma0, A, C, Q, R)
    X = cell(T,1); % states
    Y = cell(T,1); % observables
    k = size(A,1);
    d = size(C,1);
    for trial = 1:T
        trialX = zeros(k,t(trial));
        trialY = zeros(d,t(trial));
        x = mvnrnd(mu0, sigma0).';
        y = mvnrnd(C * x, R).';
        trialX(:,1) = x;
        trialY(:,1) = y;
        for time = 2:t(trial)
            x = mvnrnd(A * x, Q).';
            y = mvnrnd(C * x, R).';
            trialX(:,time) = x;
            trialY(:,time) = y;
        end
        X{trial} = trialX;
        Y{trial} = trialY;
    end
end

function res = zScore(Y)
    trials = numel(Y);
    res = cell(trials,1);
    for trial = 1:trials
        for dim = 1:size(Y{trial},1)
            y = Y{trial}(dim,:);
            res{trial}(dim,:) = (y - mean(y)) / sqrt(var(y));
        end
    end
end
