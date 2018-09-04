classdef mixHHMM < handle
    properties
        NC  % number of clusters (each cluster is defined by a HMM)
        Qdim
        dim % dimension of the state space
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        HHMMs
        pi
        
        p
        L
    end

    methods
        
        function self = mixHHMM(NC,Qdim,dim,D,obsTypes,alpha_0,Qalpha_0, Qpi0alpha_0,Aalpha_0,pi0alpha_0)
            if(isempty(obsTypes))
                obsTypes{1}.dist='mvn';
                obsTypes{1}.idx=[1:D];
            end
            
            self.NC = NC;
            self.Qdim=Qdim;
            self.dim = dim;
            self.D = D;
            for i=1:NC
                self.HHMMs{i}=HHMM(Qdim,dim,D,obsTypes,Qalpha_0, Qpi0alpha_0,Aalpha_0,pi0alpha_0);
            end
            self.obsTypes=obsTypes;
            self.pi=dists.expfam.dirichlet(NC,alpha_0);
        end
                
        function logptilde = updateassignments(self,data)
            %Also computes ELBO (L)
            logptilde=zeros(self.NC,length(data));
            for i=1:self.NC
                self.HHMMs{i}.Eloglikelihood(data);  % updates states;
                logptilde(i,:) =  cell2mat(self.HHMMs{i}.logZ); 
            end
            % add prior
            logptilde = bsxfun(@plus,logptilde,self.pi.loggeomean);
            
            % normalize
            self.p = bsxfun(@minus,logptilde,max(logptilde));            
            self.p = exp(self.p);
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            
            self.L = -self.KLqprior;
            idx=find(logptilde(:)>-Inf);
            self.L = self.L + sum(self.p(idx).*logptilde(idx));
            idx = find(self.p(:)>0);
            self.L = self.L - sum(self.p(idx).*log(self.p(idx)));
        end
        
        function L = update(self,data,modeliters)            
            self.updateassignments(data);
            L=self.L;
            for j=1:modeliters
            for i=1:self.NC
                if(sum(self.p(i,:))>1)
                    self.HHMMs{i}.updateparms(data,self.p(i,:)); 
                else
                    self.HHMMs{i}.updateparms({},0);
                end
            end
            end
            self.pi.update(sum(self.p,2));
        end
        
        function updateparms(self,data,p)
            % Assumes a particular clustering p for updating HHMMs parms.
            
            self.p=p;
            for i=1:self.NC
                if(sum(p(i,:))>1)
                    self.HHMMs{i}.update_states(data);
                    self.HHMMs{i}.updateparms(data,self.p(i,:)); 
                else
                    %Assign outliers to empty clusters
                    
                    
                end
            end
            self.pi.update(sum(self.p,2));
            
        end
        
        function res = KLqprior(self)
            res = + self.pi.KLqprior;
            for i=1:self.NC
                res = res + self.HHMMs{i}.KLqprior;
            end
        end
        
        function flatten(self,data,iters)
            self.updateassignments(data);
            
            for i=1:self.NC
                self.HHMMs{i}.Eloglikelihood;  % does not update states;
                logptilde(i,:) =  cell2mat(self.HHMMs{i}.logZ); 
            end
            ns=length(data);            
            NA=sum(self.p,2);
            [y,idx0]=sort(NA);
            self.p=zeros(size(self.p));
            for i=1:self.NC
                [y,idx]=sort(-logptilde(idx0(i),:));
                idx=idx(1:floor(length(data)/self.NC));
                logptilde(1:end,idx)=-Inf;
                self.p(idx0(i),idx)=1;
            end
            idx=find(sum(self.p,1)==0);
            self.p(1:self.NC,idx)=1/self.NC;
            self.pi.alpha=self.pi.alpha_0;
            [m,z]=max(self.p);
            for i=1:self.NC
            for j=1:iters
                self.HHMMs{i}.update(data(z==i));
            end
            end
        end
        
        function initialize(self,data,iters,z)
            for i=1:length(data)
                len(i)=size(data{i},2);
            end
            T=min(len);
            idx=[];
            for i=1:length(self.obsTypes)
                idx=[idx,self.obsTypes{i}.idx];
            end
            
            fprintf('Initializing...')
            for i=1:length(data)
                temp=data{i}(idx,end-T+1:end);               
                pruneddata(i,:) = temp(:);
            end
            if(~exist('z','var'))
                z = kmeans(pruneddata,self.NC);
            end
            for i=1:self.NC
                NA(i,1) = sum(z==i);
            end
            self.pi.update(NA);
            
            for j=1:iters
                for i=1:self.NC
                    self.HHMMs{i}.update(data(z==i));
                end
            end
            
            fprintf(['done.\n'])
            
        end
        
        function fillunused(self,data,neff,iters)
            idx=find(sum(self.p')<1);
            if(isempty(idx))
                return
            end
            fprintf(['Filling ',int2str(length(idx)),' unused clusters\n'])
            for i=1:self.NC
                fitq(i,:) = cell2mat(self.HHMMs{i}.logZ);
            end
            m=max(fitq);
            [m,didx]=sort(-m);
            datatemp = data(didx(1:length(idx)));
            for i=1:length(idx)
                for j=1:iters
%                    self.HHMMs{idx(i)}.update(datatemp((i-1)*neff+1:i*neff));
                    self.HHMMs{idx(i)}.update(datatemp(repmat(i,[neff,1])));
                end
            end
            self.pi.alpha=self.pi.alpha_0;
        end
        
    end
end