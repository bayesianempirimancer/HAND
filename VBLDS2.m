classdef VBLDS2 < handle
    % Bayesian linear dynamical system
    
    % x_t+1 ~ N(x_t+1 | Ax_t, Q)
    % y_t ~ N(y_t | Cx_t, R)
    % x_0 ~ N(x_0 | mu0, sigma0)    
    
    % parms = {A, C, Q, R}
    
    % p(A, Q) = matrix normal wishart
    % p(C, R) = matrix normal wishart
    
    % inference done via VBEM, referenced from Beal 2003    
    
    properties
        
        T % num trials
        t % trial lengths
        iters % iters spent fitting
        
        % Parameters
        k % dim of state space
        d % dim of obs. space
                
        A % state dynamics matrix + state noise
        C % emission matrix + obs. noise
        
        mu0 % auxiliary zeroth state mean
        invSigma0 % auxiliary zeroth state precision     
        
        % ELBO
        
        L
        dLs
        Ls
        lnpY % marginal loglike
        
        % Latents
        % indexing is a major pain here bc of auxiliary state
        % see notes throughout
        % to get alpha_t, access alpha(t)
        % to get beta_t, access beta(t+1)
        % to get X_t, access X(t+1)
        
        X_means % X{trial} = [k, time]
        X_covars % X{trial} = [k, k, time]
        X_crossts % X{trial} = [k, k, time] -- covars on p(x_t, x_t+1 | Y)     
        
        % filtered dists. on x
        alpha_means 
        alpha_Ps
        
        % p(x_t | y_t+1...y_T)
        beta_means 
        beta_Ps
        
    end
    
    methods
        
        function self = VBLDS2(k, d)
            % constructor
            
            self.L = -inf;
            self.iters = 0;
            
            self.k = k;
            self.d = d;
                        
            self.A = dists.expfam.matrixnormalWishart(...
                zeros(k,k),...
                eye(k),... % U = RV for dynamics noise
                eye(k)); % this is not an RV
                
                
            self.C = dists.expfam.matrixnormalWishart(...
                .5 * eye(d),...
                eye(d),... % U = RV for obs. noise
                eye(k)); % this is not an RV            
                                    
            self.mu0 = zeros(k,1);
            self.invSigma0 = 1e-7 * eye(k);
            
        end
        
        function fit(self, Y, iters)
            % fit function
            
            self.T = size(Y,1);
            for trial = 1:self.T
                self.t(trial) = size(Y{trial},2);
            end
            
            i = 1;
            while (i <= iters)
                                
                updateLatents(self,Y);
                disp("updated latents, iter " + (self.iters + i));
                
                updateELBO(self, self.iters + i); 
                
                updateParms(self,Y);
                disp("updated parms, iter " + (self.iters + i));
       
                i = i + 1;                              
            end
            
            self.iters = self.iters + i - 1;
            
        end
        
        function updateLatents(self, Y)
           % there are some redundant computations here 
            
           % E step
           % self.alpha are filtered state dists.
           % self.beta are p(y_t+1...y_T | x_t)
           % self.X are smoothed state dists.           
                     
           self.lnpY = 0; % for calculation of F / ELBO
           
           % parameter expectations / needed things
           ATinvQ = self.A.EXTinvU;
           CTinvR = self.C.EXTinvU;
           ATinvQA = self.A.EXTinvUX;
           CTinvRC = self.C.EXTinvUX;
           invQ = self.A.EinvU;
           invR = self.C.EinvU;           
                                 
           for trial = 1:self.T
               
               n = self.t(trial);
               
               % need these for both halves
               % to get sigmaStar_t, access SigmaStars(t+1)
               sigmaStars = zeros(self.k, self.k, n); 
               
               % Forward
               for time = 1:n                                                     
                   
                   y_t = Y{trial}(:,time);
                   
                   % initial step relies on auxiliary x0
                   % invSigmaStar is always from previous step / init.
                   if time == 1
                       mu_prev = self.mu0;
                       invSigma_prev = self.invSigma0;                      
                       invSigmaStar = self.invSigma0 + ATinvQA;
                       sigmaStars(:,:,time) = inv(invSigmaStar);
                   else % recursion
                       invSigma_prev = invSigma_t;       
                       mu_prev = mu_t;                        
                       invSigmaStar = invSigma_t + ATinvQA;
                       sigmaStars(:,:,time) = inv(invSigmaStar);                          
                   end                                    
                   
                   % alphas
                   invSigma_t = invQ + CTinvRC - ATinvQ' / invSigmaStar * ATinvQ;
                   self.alpha_Ps{trial}(:,:,time) = invSigma_t;
                   
                   mu_t = invSigma_t \ (CTinvR * y_t + ATinvQ' / invSigmaStar * invSigma_prev * mu_prev);  
                   self.alpha_means{trial}(:,time) = mu_t;                   
                                               
                   % E log like of y_time | y_1...y_time-1                    
                   loglike = self.d * log(2*pi) - self.C.ElogdetinvU - log(det(invSigma_prev / invSigmaStar / invSigma_t)) ...
                       + mu_prev' * invSigma_prev * mu_prev - mu_t' * invSigma_t * mu_t + y_t' * invR * y_t ...
                       - mu_prev' * invSigma_prev / invSigmaStar * invSigma_prev * mu_prev;
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
                                            
               for time = fliplr(1:n)
                   
                   y = Y{trial}(:,time);
                   
                   invPsiStar = invQ + CTinvRC + invPsi_t;               
                   invPsi_after = invPsi_t;
                   eta_after = eta_t;                           
                   
                   % note invPsiStar is always from previous step (later in
                   % time)

                   % betas
                   invPsi_t = ATinvQA - ATinvQ / invPsiStar * ATinvQ';
                   self.beta_Ps{trial}(:,:,time) = invPsi_t;
                   
                   eta_t = invPsi_t \ ATinvQ / invPsiStar * (CTinvR * y + invPsi_after * eta_after);
                   self.beta_means{trial}(:,time) = eta_t;                      
                   
                   % crosst covars
                   sigmaStar = sigmaStars(:,:,time);
                   invPart = (invQ + CTinvRC + invPsi_after - ATinvQ' * sigmaStar * ATinvQ);
                   upsilon_crosst = sigmaStar * ATinvQ / invPart;
                   self.X_crossts{trial}(:,:,time) = upsilon_crosst;                              
                   
               end                       
               
               % combine alphas and betas
               % X: to get t, access t+1
               
               for time=0:n          
                   
                   % alpha message
                   if time == 0
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
            
            % check to see the signs on these. KLs should decrease bound
            % i think in mixHHMM the pi.KLqprior should be subtracted?
%             self.A.KLqprior
%             self.C.KLqprior
            newL = - self.A.KLqprior - self.C.KLqprior + self.lnpY;
            if i > 1
                self.dLs(i-1) = newL - self.L;
            end
            self.L = newL;
            self.Ls(i) = self.L;        
        end
        
        function updateParms(self, Y)
           % M step
           
           % prep for computing sufficient stats
%            bigY = [Y{:}]';
%            bigX = [self.X_means{:}]';
%            bigU = cat(3,self.X_covars{:});
%                                           
%            % sufficient stats
% 
%            % XX blows up over iters but it's because of the X'*X, not the
%            % sum of upsilons
%            XX = bigX' * bigX + sum(bigU,3); 
%            YX = bigY' * bigX;
%            YY = bigY' * bigY;
           
%            aux = zeros(self.k, self.k);
           XXnoAux = zeros(self.k,self.k);
           XXnoEnds = zeros(self.k,self.k);
           XXcrosst = zeros(self.k,self.k);
           YX = zeros(self.d,self.k);
           YY = zeros(self.d,self.d);
%            YY = bigY'*bigY;
                      
           for trial = 1:self.T
               
                              
               for time = 1:self.t(trial)
                   % this works because there are t crossts and t+1 x's
                   y_t = Y{trial}(:,time);
                   omega_t = self.X_means{trial}(:,time);
                   omega_t1 = self.X_means{trial}(:,time+1);
                   upsilon_t = self.X_covars{trial}(:,:,time);
                   upsilon_crosst = self.X_crossts{trial}(:,:,time);                   
                   
                   XXnoEnds = XXnoEnds + upsilon_t + omega_t * omega_t';                   
                   XXcrosst = XXcrosst + upsilon_crosst + omega_t * omega_t1';
                   YX = YX + y_t * omega_t1';
                   YY = YY + y_t * y_t';
                                      
               end
               
               for time = 2:self.t(trial)+1
                   omega_t = self.X_means{trial}(:,time);                   
                   upsilon_t = self.X_covars{trial}(:,:,time);
                   
                   XXnoAux = XXnoAux + upsilon_t + omega_t * omega_t';
               end
               
           end        
           
%            XXnoEnds = makePSD(XXnoEnds)
%            XXcrosst
%            XXnoAux = makePSD(XXnoAux)
%            YX = YX
%            YY = makePSD(YY)
                                                        
           % update           
           N = sum(self.t);
           self.A.updateSS(XXnoEnds/N, XXcrosst/N, XXnoAux/N, N);
           N = sum(self.t);
           self.C.updateSS(XXnoAux/N, YX/N, YY/N,N);                    
           
           newMu0 = zeros(self.k,1);                                
           for trial = 1:self.T
               newMu0 = newMu0 + self.X_means{trial}(:,1);
           end
           self.mu0 = newMu0 / self.T;
           
           newInvSigma0 = zeros(self.k,self.k);           
           for trial = 1:self.T
               dif = self.mu0 - self.X_means{trial}(:,1);
               newInvSigma0 = newInvSigma0 + self.X_covars{trial}(:,:,1) + dif * dif';
           end                                
           self.invSigma0 = newInvSigma0 / self.T;           
           
        end
        
        function plotLs(self)
           % plots L over iterations
           plot(1:length(self.Ls),self.Ls)
           title("L by iteration")
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
   if rank(V) < size(V,2)
      disp("singular matrix");
   end
   if ~(isPSD(V))
      [P,D] = eig(V);
      D = diag(D);                      
      D(D<eps) = eps;                      
      D = diag(D);
      res = P * D * P';
   end    
end

