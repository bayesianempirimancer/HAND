classdef SHHMM < handle
    properties
        NC  % number of clusters (each cluster is defined by a HMM)
        dim % dimension of the state space
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        HMMs
        pi
        
        logptilde
        p
        NA
        L
    end

    methods
        function self = SHHMM(NC,dim,D,obsTypes,alpha_0,Aalpha_0,pi0alpha_0)
            if(~exist('obsTypes','var'))
                obsTypes{1}.dist='mvn';
                obsTypes{1}.idx=[1:D];
            end
            if(~exist('alpha_0','var'))
                alpha_0=ones(NC,1);
            end
            if(~exist('Aalpha_0','var'))
                Aalpha_0=ones(dim,1);
            end            
            if(~exist('pialpha_0','var'))
                pialpha_0=ones(dim,1);
            end
            
            self.NC = NC;
            self.dim = dim;
            self.D = D;
            for i=1:NC
                self.HMMs{i}=HMM(dim,D,obsTypes,Aalpha_0,pi0alpha_0);
            end
            self.obsTypes=obsTypes;
            self.pi=dists.expfam.dirichlet(dim,alpha_0);
        end
        
        function L = update(self,data)
            self.logptilde=zeros(self.NC,length(data));
            for i=1:self.NC
                self.HMMs{i}.Eloglikelihood(data);
                self.logptilde(i,:) =  cell2mat(self.HMMs{i}.logZ); 
            end
            % add prior
            self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);
            
            % normalize
            self.p = bsxfun(@minus,self.logptilde,max(self.logptilde));            
            self.p = exp(self.p);
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            self.NA = sum(self.p,2);
            
            self.L = -self.KLqprior;
            self.L = self.L + sum(self.p(:).*self.logptilde(:));
            idx = find(self.p(:)>0);
            self.L = self.L - sum(self.p(idx).*log(self.p(idx)));
                        
            for i=1:self.NC
                self.HMMs{i}.updateparms(data,self.p(i,:)); 
            end
            self.pi.update(self.NA);
            L = self.L;
        end
        
        function res = KLqprior(self)
            res = + self.pi.KLqprior;
            for i=1:self.NC
                res = res + self.HMMs{i}.KLqprior;
            end
        end
        
    end
end