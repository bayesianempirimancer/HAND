%% overview

% script for synthetic LDS data generation
% and fitting VBLDS

% model:
% x_t+1 ~ N(x_t+1 | Ax_t + Bu_t + b1, Q)
% y_t ~ N(y_t | Cx_T + Dy_prev + b2, R)

%% data synthesis 
rng(2018)

mu0 = [1; 1];
sigma0 = .01 * eye(2);

A = .3 * eye(2);

B = .6 * eye(2);

b1 = [.01; .02];

C = [1 0;
    0 1;
    .5 .5];

h = 3;
d = size(C,1);
D = .1 * ones(d,d*h);

b2 = [.05; .02; .1];

Q = .5 * eye(size(A,1));
R = .5 * eye(size(C,1));

T = 10; % number trials
t = 100 * ones(T,1); % trial lengths

% data stored in X{trial} = [dim, time] format
[X, U, Y] = generate(T, t, mu0, sigma0, A, B, b1, C, D, b2, Q, R);

Xtrue = [X{:}]';

%% model fitting
rng(2018)
    
k = size(A,2);
u = size(B,2);
d = size(Y{1},1);
h = size(D,2) / d;
T = size(Y,1);
model = VBLDSinputs(k, u, d, h);    

iters = 20;
% Y2 = zScore(Y);
model.fit(U,Y,iters)


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
function [X, U, Y] = generate(T, t, mu0, sigma0, A, B, b1, C, D, b2, Q, R)
    X = cell(T,1); % states
    U = cell(T,1); % inputs
    Y = cell(T,1); % observables
    k = size(A,1);
    d = size(C,1);
    dh = size(D,2);
    for trial = 1:T
        trialX = zeros(k,t(trial));
        trialU = zeros(2,t(trial));
        trialY = zeros(d,t(trial));        
        y_his = .01 * ones(dh,1); % a burn in of sorts
        x = mvnrnd(mu0, sigma0).';
        u = vertcat(sin(1/10),cos(1/10));
        y = mvnrnd(C * x + D * y_his + b2, R).';
        trialX(:,1) = x;
        trialU(:,1) = u;
        trialY(:,1) = y; 
        for time = 2:t(trial)
            y_his = vertcat(y_his(d+1:end,1), y);
            u = vertcat(sin(time/10),cos(time/10));
            x = mvnrnd(A * x + B * u + b1, Q).';
            y = mvnrnd(C * x + D * y_his + b2, R).';
            trialX(:,time) = x;
            trialU(:,time) = u;
            trialY(:,time) = y;
        end
        X{trial} = trialX;
        U{trial} = trialU;
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
