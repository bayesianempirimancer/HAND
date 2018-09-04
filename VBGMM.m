classdef VBGMM < handle

    properties
        % Parameters
        k % number of clusters
        d % dimension of the observations
        
        L % lower bound
        
        a % hyperparam for pi
        
        pi % dirichlet over clusters
        
        dist % distributions
        
        p  % assignment probabilities (n x k)
        logptilde
        nk % number of assignments to each cluster
        
        ObsTypes
              
    end
    
    methods
        function self = VBGMM(k,d)
            self.k = k;
            self.d = d;
            
            self.L = -Inf;
            
            self.a = 1/self.k;
            
            self.pi=dists.expfam.dirichlet(self.k,self.a*ones(1,self.k));
            
            self.ObsTypes{1}.dist='mvn';
            self.ObsTypes{1}.idx=(1:d);
            
            for i= 1:self.k
                self.dist{i}=dists.GPEF(self.ObsTypes);
            end
            
    
        end
        
        
        function fit(self,X,iters)


            i = 0;
            while (i < iters)
                
                updateParams(self,X);
                updateLatents(self,X);
                updateL(self,X);
                self.plotclusters(X,1)
            
                
                i = i + 1;
                
            end
            
            self.L
            sum(self.p,1) / sum(sum(self.p,1))
        end
        
        
        function updateLatents(self,X)
        
            
        self.logptilde=zeros(size(X,1),self.k);
        
        for i = 1:self.k
            self.logptilde(:,i) = self.dist{i}.Eloglikelihood(X);
        end
        self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);

        self.p = bsxfun(@minus,self.logptilde,max(self.logptilde')');            
        
        
        self.p = exp(self.p);
        self.p = bsxfun(@rdivide, self.p, sum(self.p, 2));
        
        self.nk = sum(self.p, 1);
        
        end
        
        
        function updateParams(self,X)
            if(isempty(self.p))
                self.updateLatents(X);
            end
            
            self.pi.update(self.nk);
            
            for i = 1:self.k
                self.dist{i}.update(X,self.p(:,i));
            end
            
        end
        
        function updateL(self, X)
            
            self.L = - self.KLqprior;
            for i = 1:self.k
                self.L = self.L + sum(self.dist{i}.Eloglikelihood(X));
            end
            
        end
        
 
        function KL = KLqprior(self)            

            KL = self.pi.KLqprior;            
            for i=1:self.k
                KL = KL + self.dist{i}.KLqprior;
            end
                        
        end
        
        
        function plotclusters(self,X,fighandle)
            figure(fighandle) 


            [temp,idx] = max(self.p');
            clusters=unique(idx);
            cc0=jet(length(clusters));
            for i=1:length(clusters)
                cc(clusters(i),:)=cc0(i,:);
            end

            scatter(X(:,1),X(:,2),2*ones(size(X,1),1),cc(idx,:))

            title(strcat('<ELBO> = ',num2str(self.L/size(X,1))))
        
        
        end
        
        
        
        
        
    end
    
end

