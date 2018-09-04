classdef sharedHHMM < handle
    properties
        % Two level heirarchal hidden markov model.

        Qdim % number of high level states
        Q % transition probabilities at highest level
        Qpi0 % initial high level state distribution

        dim % size of low level state space
        A   % A{i} is the transition probability matrix between low level 
            % states when in high level state i
        pi0 % pi0{i} is the initial state distribution after a highlevel transition
            % to state i occurs. 
        Aidx % relates Q to HMMequiv.  The idea is to give HMMequiv.A block 
             % structure that groups Q states, i.e. diag(A{k}*Q(k,k)) 
             
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        obsModels 
        
        HMMequiv
        p
        Qp
        Exx
        logZ
        L
    end

    methods
        function self = sharedHHMM(Qdim,dim,D,obsTypes,Qalpha_0, Qpi0alpha_0,Aalpha_0,pi0alpha_0)
            self.Qdim = Qdim;
            self.dim = dim;
            self.D = D;
            self.Q=dists.transition(dim,Qalpha_0);
            self.Qpi0=dists.expfam.dirichlet(dim,Qpi0alpha_0);

            for i = 1:Qdim
                self.Aidx{i}=[1:dim]+(i-1)*dim;
                self.A{i}  = dists.transition(dim,Aalpha_0);
                self.pi0{i}= dists.expfam.dirichlet(dim,pi0alpha_0);
            end
            
            if(isempty(obsTypes))
                fprintf('Defaulting to normally distributed observations.\n');
                self.obsTypes{1}.dist = 'mvn';
                self.obsTypes{1}.idx = [1:D];
            else
                self.obsTypes = obsTypes;
            end
            
            for i=1:dim
                self.obsModels{i}=dists.GPEF(self.obsTypes);
            end
            
            self.HMMequiv=HMM(self.Qdim*self.dim,D,obsTypes,Aalpha_0,pi0alpha_0);
            
        end
        
        function res = obsloglike(self,data)
            % outputs a cell array of  T x dim matrix of likelihoods 
            % assuming standard data of the form data{trials}(D , T)
            res={};
            for i=1:numel(data)
                res{i}=zeros(self.dim,size(data{i},2));
                for k=1:self.dim
                    res{i}(k,:) = self.obsModels{k}.Eloglikelihood(data{i}')';  
                    % these primes needs to be fixed in Eloglikelihood computation
                end
                res{i}=repmat(res{i},self.Qdim,1);
            end
        end
        
        function update_states(self,data)
            
            obsloglike = self.obsloglike(data);
            
            logQ = self.Q.loggeomean;
            logQpi0 = self.Qpi0.loggeomean;
            logpi0 = zeros(self.Qdim*self.dim,1);
            for k=1:self.Qdim
                logA(self.Aidx{k},self.Aidx{k}) ...
                    = logQ(k,k)+self.A{k}.loggeomean;
                logpi0(self.Aidx{k},1) = logQpi0(k,1) + self.pi0{k}.loggeomean;
                
                for l=[1:k-1,k+1:self.Qdim]
                    logA(self.Aidx{k},self.Aidx{l}) ...
                        = logQ(k,l)+repmat(self.pi0{l}.loggeomean',self.dim,1);
                end
            end

            for i=1:numel(data) % can be parallelized 
                [p,Exx,logZ] = self.HMMequiv.forwardbackward(obsloglike{i},logA,logpi0);
                self.p{i} = p;
                self.Exx{i} = Exx;
                self.logZ{i} = logZ;                             
                self.Qp{i}=zeros(self.Qdim,size(p,2));
                for j=1:self.Qdim
                    self.Qp{i}(j,:)=sum(p(self.Aidx{j},:),1);
                end
            end
            self.p = self.p(1:length(data));
            self.Qp = self.Qp(1:length(data));
            self.Exx = self.Exx(1:length(data));
            self.logZ = self.logZ(1:length(data));
            
        end

        function L = update(self,data)
            self.update_states(data);  
            self.L = sum(cellfun(@sum,self.logZ)) - self.KLqprior;
                %SExx and SEx0 are stored in HMMequiv format                
            L=self.L;
            
            SExx = zeros(self.HMMequiv.dim,self.HMMequiv.dim);
            SEx0 = zeros(self.HMMequiv.dim,1);
            for i=1:numel(data);
                SExx = SExx + self.Exx{i};
                SEx0 = SEx0 + self.p{i}(:,1);
            end
            
            for k=1:self.Qdim
                temp0(k)=sum(SEx0(self.Aidx{k}));
            for l=1:self.Qdim
                temp(k,l) = sum(sum(SExx(self.Aidx{k},self.Aidx{l})));
            end
            end
            self.Q.update(temp);
            self.Qpi0.update(temp0');
            
            for k=1:self.Qdim
                self.A{k}.update(SExx(self.Aidx{k},self.Aidx{k}));
                temp = zeros(self.dim,self.dim);
                for l=[1:k-1,k+1:self.Qdim]
                    temp = temp + SExx(self.Aidx{l},self.Aidx{k});
                end
                temp=sum(temp,1);
                self.pi0{k}.update(SEx0(self.Aidx{k})+temp');
            end
            
            datacat = [data{:}];
            pcat = [self.p{:}];
            
            for k=1:self.dim
                pA = zeros(1,size(pcat,2));
                for j=1:self.Qdim
                    pA = pA + pcat(self.Aidx{j}(k),:);
                end
                
                if(sum(pA)>1)
                    self.obsModels{k}.update(datacat',pA');
                else
                    self.obsModels{k}.update([],0);
                end
            end
        end
        
        function updateparms(self,data,p) 
            % assumes states are up to date.  
            if(~exist('p','var'))
                p=ones(numel(data),1); 
            else
                n=sum(p);
            end
            SExx = zeros(self.HMMequiv.dim,self.HMMequiv.dim);
            SEx0 = zeros(self.HMMequiv.dim,1);
            pcat=cell(size(data));
            for i=1:numel(data);
                SExx = SExx + self.Exx{i}*p(i);
                SEx0 = SEx0 + self.p{i}(:,1)*p(i);
                pcat{i} = self.p{i}*p(i);
            end
            
            for k=1:self.Qdim
                temp0(k)=sum(SEx0(self.Aidx{k}));
            for l=1:self.Qdim
                temp(k,l) = sum(sum(SExx(self.Aidx{k},self.Aidx{l})));
            end
            end
            self.Q.update(temp);
            self.Qpi0.update(temp0');
            
            for k=1:self.Qdim
                self.A{k}.update(SExx(self.Aidx{k},self.Aidx{k}));
                temp = zeros(self.dim,self.dim);
                for l=[1:k-1,k+1:self.Qdim]
                    temp = temp + SExx(self.Aidx{l},self.Aidx{k});
                end
                temp=sum(temp,1);
                self.pi0{k}.update(SEx0(self.Aidx{k})+temp');
            end
            
            datacat = [data{:}];
            pcat = [self.p{:}];
            
            for k=1:self.dim
                pA = zeros(1,size(pcat,2));
                for j=1:self.Qdim
                    pA = pA + pcat(self.Aidx{j}(k),:);
                end
                
                if(sum(pA)>1)
                    self.obsModels{k}.update(datacat',pA');
                else
                    self.obsModels{k}.update([],0);
                end
            end
    
        end
           
        function init_obsmodels(self,data)
           datacat = [data{:}];
           idx=[];
           for i=1:length(self.obsTypes);
               idx=[idx,self.obsTypes{i}.idx];
           end
           z=kmeans(datacat(idx,:)',self.dim);
           for i=1:self.dim
               self.obsModels{i}.update(datacat(:,z==i)',10*ones(sum(z==i),1)/(1+sum(z==i)));
           end            
        end
        
        function plotAs(self)
            for i=1:self.dim
                mu(i,:)=self.obsModels{i}.mean;
            end
            mu=mu(:,3:4);
            figure
            px=ceil(sqrt(self.Qdim));
            py=ceil(self.Qdim/px);
            temp = diag(self.Q.mean);
            dur = ceil(1./(1-temp));
            
            for i=1:self.Qdim
                cs=self.pi0{i}.mean'*self.A{i}.mean^(dur(i));
                subplot(px,py,i), scatter(mu(:,2),mu(:,1),100*cs)
            end
            figure
            for i=1:self.Qdim
                subplot(px,py,i), imagesc(self.A{i}.mean)
            end
            
            
        end
        
        function res = Eloglikelihood(self,data)
            if(~exist('data','var'))
                res = self.logZ;
            else
                self.update_states(data);
                res = self.logZ;
            end
        end
             
        function res = KLqprior(self)
            res = self.Q.KLqprior + self.Qpi0.KLqprior;
            for i=1:self.dim
                res = res + self.obsModels{i}.KLqprior;
            end
            for i=1:self.Qdim
                res = res + self.A{i}.KLqprior;
                res = res + self.pi0{i}.KLqprior;
            end
        end

    end
end