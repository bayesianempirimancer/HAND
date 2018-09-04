classdef VBLDSinputs < handle
    % Bayesian linear dynamical system
    
    % x_t+1 ~ N(x_t+1 | Ax_t + Bu_t + b1, Q)
    % y_t ~ N(y_t | Cx_t + Dy_prev + b2, R)
    % x_0 ~ N(x_0 | mu0, sigma0)    
        
    % inference done via VBEM, referenced from Beal 2003     
    
    properties
        
        T % num trials
        t % trial lengths
        iters % iters spent fitting
        obs % data
        
        % Parameters
        k % dim of state space
        u % dim of state inputs
        d % dim of obs. space
        h % number of past ys in autoregression (history)
                
        A % [state dynamics matrix, input dynamics, bias] and state noise
        C % [emission matrix, autoregression, bias] and obs. noise
        
        mu0 % auxiliary zeroth state mean
        invSigma0 % auxiliary zeroth state precision     
        
        % ELBO
        
        L
        dLs
        Ls
        lnpY
        
        % Latents
        % indexing is a major pain here bc of auxiliary state
        % see notes throughout
        % to get alpha_t, access alpha(t)
        % to get beta_t, access beta(t+1)
        % to get X_t, access X(t+1)
        
        X_means % X{trial} = [k, time]
        X_covars % X{trial} = [k, k, time]
        X_crossts % X{trial} = [k, k, time] -- covars on p(x_t-1, x_t | Y)     
        
        % filtered dists. on x
        alpha_means 
        alpha_Ps
        
        % p(x_t | y_t+1...y_T)
        beta_means 
        beta_Ps
        
        % for debugging, these store the values at each iteration
        % works across multiple calls of .fit
        lnpYs
        AKLs
        CKLs
        R2s
        
        % idx
        Aidx
        Bidx
        Cidx
        Didx
        
        % transformed Chat
        Ctrue
        
    end
    
    methods
        
        function self = VBLDSinputs(k,u,d,h)
            % constructor
            
            self.L = -inf;
            self.iters = 0;
            
            self.k = k;
            self.u = u;
            self.d = d;
            self.h = h;
                        
            self.A = dists.expfam.matrixnormalWishart(...
                zeros(k,k+u+1),...
                eye(k),... 
                eye(k+u+1));                 
                
            self.C = dists.expfam.matrixnormalWishart(...
                zeros(d,k+d*h+1),...
                eye(d),... 
                eye(k+d*h+1));
                                    
            self.mu0 = zeros(k,1);
            self.invSigma0 = 1e-7 * eye(k);    
            
        end
        
        function fit(self, U, Y, iters)
            
            self.T = size(Y,1);
            for trial = 1:self.T
                self.t(trial) = size(Y{trial},2);
            end
            self.obs = Y;
            
            i = 1;
            while (i <= iters)
                                
                updateLatents(self,U,Y);                
                updateELBO(self, self.iters + i);                 
                updateParms(self,U,Y);
                
                if mod(i, iters/10) == 0
                    fprintf('iters done: %d\n', i);
                end

                % UNCOMMENT this to see R2s over iters
%                 self.R2s(:,self.iters + i) = self.R2;
       
                i = i + 1;                              
            end
            
            self.iters = self.iters + i - 1;
                        
        end
        
        function updateLatents(self, U, Y)
           % NOTE there are some redundant matrix computations here 
            
           % self.alpha are filtered state dists.
           % self.beta are p(y_t+1...y_T | x_t)
           % self.X are smoothed state dists.           
                     
           self.lnpY = 0;
                      
           % ids
           self.Aidx = 1:self.A.p < self.k+1;
           self.Bidx = ~self.Aidx;
           self.Cidx = 1:self.C.p < self.k+1;
           self.Didx = ~self.Cidx;    

           
           % parameter expectations broken into blocks
           ABQ = self.A.EXTinvU;
           ATinvQ = ABQ(self.Aidx,:);
           BTinvQ = ABQ(self.Bidx,:);
           
           CDR = self.C.EXTinvU;
           CTinvR = CDR(self.Cidx,:);
           DTinvR = CDR(self.Didx,:);
                      
           ABQAB = self.A.EXTinvUX;
           ATinvQA = ABQAB(self.Aidx,self.Aidx); 
           ATinvQB = ABQAB(self.Aidx,self.Bidx);
           
           CDRCD = self.C.EXTinvUX;
           CTinvRC = CDRCD(self.Cidx,self.Cidx);
           CTinvRD = CDRCD(self.Cidx,self.Didx);
           DTinvRD = CDRCD(self.Didx,self.Didx);
           
           invQ = self.A.EinvU;           
           invR = self.C.EinvU;                                  
                                 
           for trial = 1:self.T
               
               n = self.t(trial);
               
               % need these for both halves
               % to get sigmaStar_t, access SigmaStars(t+1)
               invSigmaStars = zeros(self.k, self.k, n); 
               
               % Forward
               for time = self.h+1:n                                                     
                   
                   y_t = Y{trial}(:,time);
                   y_his = Y{trial}(:,time-self.h:time-1);
                   y_his = vertcat(reshape(y_his,self.h*self.d,1), 1);  
                   u_t = vertcat(U{trial}(:,time),1);
                   
                   % initial step relies on auxiliary x0
                   % invSigmaStar is always from previous step / init.
                   if time == self.h+1
                       mu_prev = self.mu0;
                       invSigma_prev = self.invSigma0;                      
                       invSigmaStar = self.invSigma0 + ATinvQA;
                       invSigmaStars(:,:,time) = invSigmaStar;
                   else % recursion
                       invSigma_prev = invSigma_t;       
                       mu_prev = mu_t;                        
                       invSigmaStar = invSigma_t + ATinvQA;
                       invSigmaStars(:,:,time) = invSigmaStar;                          
                   end                                    
                   
                   % alphas
                   invSigma_t = invQ + CTinvRC - ATinvQ' / invSigmaStar * ATinvQ;
                   self.alpha_Ps{trial}(:,:,time) = invSigma_t;
                   
                   mu_t = invSigma_t \ (CTinvR * y_t - CTinvRD * y_his + ATinvQ' / invSigmaStar * invSigma_prev * mu_prev ...
                       + (BTinvQ' - ATinvQ' / invSigmaStar * ATinvQB) * u_t);  
                   self.alpha_means{trial}(:,time) = mu_t;                   
                                               
                   % E log like of y_time | y_1...y_time-1    
                   temporary = invSigma_prev * mu_prev - ATinvQB * u_t;
                   loglike = self.d * log(2*pi) - self.C.ElogdetinvU - log(det(invSigma_prev / invSigmaStar / invSigma_t)) ...
                       + mu_prev' * invSigma_prev * mu_prev - mu_t' * invSigma_t * mu_t + y_t' * invR * y_t ...
                       - 2 * y_t' * DTinvR' * y_his + y_his' * DTinvRD * y_his ...
                       - (temporary)' / invSigmaStar * temporary;
                   loglike = -1/2 * loglike;   
                   self.lnpY = self.lnpY + loglike;
                                       
               end
               % Backward
               % beta: to get t, access t+1
               % note that crosst covars are calculated here                              
               % crosst: to get cross_t,t+1 access t+1
               
               % init
               invPsi_t = zeros(self.k, self.k);               
               self.beta_Ps{trial}(:,:,n+1) = invPsi_t;

               eta_t = zeros(self.k,1);
               self.beta_means{trial}(:,n+1) = eta_t;                                       
                                            
               for time = fliplr(self.h+1:n)
                   
                   % y_t+1 used for Psi_t
                   y_t1 = Y{trial}(:,time);
                   y_his = Y{trial}(:,time-self.h:time-1);
                   y_his = vertcat(reshape(y_his,self.h*self.d,1), 1);  
                   u_t1 = vertcat(U{trial}(:,time),1);
                                      
                   invPsiStar = invQ + CTinvRC + invPsi_t;               
                   invPsi_after = invPsi_t;
                   eta_after = eta_t;                           
                   
                   % note invPsiStar is always from previous step (later in
                   % time)

                   % betas
                   invPsi_t = ATinvQA - ATinvQ / invPsiStar * ATinvQ';
                   self.beta_Ps{trial}(:,:,time) = invPsi_t;
                   
                   eta_t = invPsi_t \ (-ATinvQB * u_t1 + ATinvQ / invPsiStar * (CTinvR * y_t1 ...
                       + BTinvQ' * u_t1 - CTinvRD * y_his + invPsi_after * eta_after));
                   self.beta_means{trial}(:,time) = eta_t;                      
                   
                   % crosst covars
                   % i think this is wrong because we want <(x_t+1 - omega_t+1) * (x_t - omega_t)'>
                   % except whatever tweaks i tried made things worse
                   invSigmaStar = invSigmaStars(:,:,time);
                   invPart = (invQ + CTinvRC + invPsi_after - ATinvQ' / invSigmaStar * ATinvQ);
                   upsilon_crosst = invSigmaStar \ ATinvQ / invPart;
                   self.X_crossts{trial}(:,:,time) = upsilon_crosst;  
                   
               end                       
               
               % combine alphas and betas
               % X: to get t, access t+1
               
               for time = self.h:n          
                   
                   % alpha message
                   if time == self.h
                       mu_t = self.mu0;
                       invSigma_t = self.invSigma0;
                   else
                       mu_t = self.alpha_means{trial}(:,time);
                       invSigma_t = self.alpha_Ps{trial}(:,:,time);
                   end                                 
                                                       
                   % beta message
                   eta_t = self.beta_means{trial}(:,time+1);
                   invPsi_t = self.beta_Ps{trial}(:,:,time+1);                

                   % combine                   
                   invUpsilon_t = invSigma_t + invPsi_t;
                   upsilon_t = inv(invUpsilon_t);
                   self.X_covars{trial}(:,:,time+1) = upsilon_t;
                   
                   omega_t = invUpsilon_t \ (invSigma_t * mu_t + invPsi_t * eta_t);
                   self.X_means{trial}(:,time+1) = omega_t;    
                   
               end               
               
           end           
                      
        end
        
        function updateELBO(self, i)
            % update lower bound
            
            self.lnpYs(i) = self.lnpY;
            self.AKLs(i) = self.A.KLqprior;
            self.CKLs(i) = self.C.KLqprior;
                        
            newL = - self.A.KLqprior - self.C.KLqprior + self.lnpY;
            if i > 1
                self.dLs(i-1) = newL - self.L;
            end
            self.L = newL;
            self.Ls(i) = self.L;        
            
        end
        
        function updateParms(self, U, Y)
           % NOTE much of this could be vectorized                       
           
           % sufficient stats
           ABregdim = self.k + self.u + 1;
           ABin = zeros(ABregdim,ABregdim);
           ABcross = zeros(self.k,ABregdim);
           ABout = zeros(self.k,self.k);
           
           CDregdim = self.k + self.d * self.h + 1;
           CDin = zeros(CDregdim,CDregdim);
           CDcross = zeros(self.d,CDregdim);
           CDout = zeros(self.d,self.d);          
                      
           for trial = 1:self.T                              
                              
               for time = self.h+1:self.t(trial)                   
                   % to get omega_t+1, access X_means(t)
                   % to get upsilon_t+1, access X_covars(t)
                   % to get upsilon_t,t+1 access X_covars(t)
                   
                   % grab all the things                   
                   omega_t = self.X_means{trial}(:,time);
                   omega_t1 = self.X_means{trial}(:,time+1);
                   u_t = U{trial}(:,time-1);
                   in = vertcat(omega_t,u_t,1);                                      
                   upsilon_t = self.X_covars{trial}(:,:,time);
                   upsilon_t(ABregdim,ABregdim) = 0; % this pads
                   upsilon_t1 = self.X_covars{trial}(:,:,time+1);
                   upsilon_crosst = self.X_crossts{trial}(:,:,time);                   
                   upsilon_crosst(:,ABregdim) = 0; % this pads
                   
                   % and add them
                   ABin = ABin + upsilon_t + in * in';                                      
                   ABcross = ABcross + upsilon_crosst + omega_t1 * in';                                      
                   ABout = ABout + upsilon_t1 + omega_t1 * omega_t1';
               end 
               
               for time = self.h+1:self.t(trial)
                                
                   % grab all the things
                   omega_t = self.X_means{trial}(:,time+1);
                   y_his = Y{trial}(:,time-self.h:time-1);
                   y_his = reshape(y_his,self.h*self.d,1);                                     
                   in = vertcat(omega_t, y_his, 1);
                   y_t = Y{trial}(:,time);                   
                   
                   % and add them
                   CDin = CDin + in * in';
                   CDcross = CDcross + y_t * in';
                   CDout = CDout + y_t * y_t';
                   
               end                                                     
               
           end                   
                                                        
           % update           
           N = sum(self.t) - self.T*self.h + self.T; % lose the history, include the aux state
           self.A.updateSS(ABin/N, ABcross/N, ABout/N, N);
           N = sum(self.t) - self.h*self.T; % lose the history
           self.C.updateSS(CDin/N, CDcross/N, CDout/N,N);                    
           
           newMu0 = zeros(self.k,1);                                
           for trial = 1:self.T
               newMu0 = newMu0 + self.X_means{trial}(:,self.h+1);
           end
           self.mu0 = newMu0 / self.T;
           
           newInvSigma0 = zeros(self.k,self.k);           
           for trial = 1:self.T
               dif = self.mu0 - self.X_means{trial}(:,1);
               newInvSigma0 = newInvSigma0 + self.X_covars{trial}(:,:,self.h+1) + dif * dif';
           end                                
           self.invSigma0 = newInvSigma0 / self.T;                        
           
        end
        
        % end of the inference methods
        
        function [pred_means, pred_covars] = getPreds(self)
           % returns mean and covar of p(newY_t | data) for all t
           pred_means = cell(self.T,1);
           pred_covars = cell(self.T,1);                      
                      
           CDmean = self.C.mean;
           R = inv(self.C.invU.mean);
                      
           for trial = 1:self.T
              for time = self.h+1:self.t(trial)
                 omega_t = self.X_means{trial}(:,time+1);
                 upsilon_t = self.X_covars{trial}(:,:,time+1);
                 dim = self.k + self.d*self.h + 1;
                 upsilon_t(dim,dim) = 0; % pad
                 y_his = self.obs{trial}(:,time-self.h:time-1);
                 y_his = vertcat(reshape(y_his,self.h*self.d,1),1); 
                 
                 CDUDC = self.C.EXAXT(upsilon_t);
                 
                 pred_means{trial}(:,time) = CDmean * vertcat(omega_t,y_his);
                 pred_covars{trial}(:,:,time) = CDUDC + R; % is this right?                 
              end
           end
           
        end
        
        function r = r(self,Xtrue)
            % Xtrue should by n x d
            % returns r from canoncorr and
            % calcs Ctrue (scale remains off right?)
            
            Xhat = zeros(sum(self.t),self.k);
            for trial = 1:self.T
                for time = self.h:self.t(trial)
                    omega_t = self.X_means{trial}(:,time+1);
                    if trial == 1
                        index = time;
                    else
                        index = sum(self.t(1:trial-1)) + time;
                    end                    
                    Xhat(index,:) = omega_t';
                end
            end   
            [a,b,r] = canoncorr(Xtrue, Xhat);
            Cmean = self.C.mean;
            self.Ctrue = Cmean(:,self.Cidx) * (b \ a)';
            
        end        
        
        function R2 = R2(self)
            % compute R2 for each observed dimension
            SSR = zeros(self.d,1);
            var = zeros(self.d,1);
            Ymean = mean([self.obs{:}],2);
            Cmean = self.C.mean;
            
            % need to take out the burned history terms from Ymean
            hist_contribution = zeros(self.d,1);
            for trial = 1:self.T
                for time = 1:self.h
                    hist_contribution = hist_contribution + self.obs{trial}(:,time);
                end
            end
            Ymean = Ymean - hist_contribution / sum(self.t);
            
            for trial = 1:self.T                
                for time = self.h+1:self.t(trial)                    
                    x = self.X_means{trial}(:,time+1);
                    y = self.obs{trial}(:,time);
                    y_his = self.obs{trial}(:,time-self.h:time-1);
                    y_his = vertcat(reshape(y_his,self.h*self.d,1),1); 
                    pred = Cmean * vertcat(x,y_his);
                    for dim = 1:self.d
                        SSR(dim) = SSR(dim) + (y(dim) - pred(dim))^2;
                        var(dim) = var(dim) + (y(dim) - Ymean(dim))^2;
                    end
                end
            end
            
            R2 = 1 - SSR./var;
            
        end
        
        function plotLs(self)
           % plots L over iterations
           figure()
           plot(self.Ls)
           title("L by iteration, input model")
        end   
        
        function plotR2s(self)
            for i = 1:self.d
               figure()
               plot(self.R2s(i,:));
               title("R2 for dim " + i);
            end
        end
        
        function negdLs(self)
           % returns the negative changes in L
           idx = self.dLs < 0;
           self.dLs(idx)
        end        
                
    end    
    
end

% helpers
function bool = isPSD(V)
    bool = all(all(V == V')) && all(eig(V) > eps);
end

function res = makePSD(V)
   res = V;
   if ~(isPSD(V))
      [P,D] = eig(V);
      D = diag(D);                      
      D(D<eps) = eps;                      
      D = diag(D);
      res = P * D * P';
   end    
end
   
function X = pad(A,dim)
    % pad a matrix with zeros so it's dim x dim
    [d1,d2] = size(A);
    X = zeros(dim,dim);
    X(1:d1,1:d2) = A;
end

