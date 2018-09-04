classdef mixSHMM < handle
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
        obsModels      % Because obsmodels are SHARED 
        HMMs
        
        pi
        
        p
        NA
        L
    end

    methods
        
        function self = mixSHMM(NC,dim,D, obsTypes, alpha_0, Aalpha_0, pi0alpha_0)
            if(isempty(obsTypes))
                obsTypes{1}.dist='mvn';
                obsTypes{1}.idx=[1:D];
                self.D = D;
            else
                self.obsTypes = obsTypes;
                self.D = 0;
                for i=1:length(obsTypes)
                    self.D = self.D + length(obsTypes{i}.idx);
                end                
            end
            for i=1:dim
                self.obsModels{i} = dists.GPEF(obsTypes);
            end
            
            self.NC = NC;
            self.dim = dim;
            
            for i=1:NC
                self.HMMs{i}=HMM(dim,D,obsTypes,Aalpha_0,pi0alpha_0);
                self.HMMs{i}.obsModels=self.obsModels;
            end
            
            self.pi=dists.expfam.dirichlet(NC,alpha_0);            
        end
                        
        function res = obsloglike(self,data)
            % outputs a cell array of  T x dim matrix of likelihoods 
            % assuming standard data of the form data{trials}(D , T)
            for i=1:numel(data)
                for k=1:self.dim
                    res{i}(k,:) = self.obsModels{k}.Eloglikelihood(data{i}')';  
                end
            end
        end

        
        function L = update(self,data,modeliters)            
            datacat = [data{:}];
            
% Update Assignments
            obslike = self.obsloglike(data);
            logptilde=zeros(self.NC,length(data));
            for i=1:self.NC
                self.HMMs{i}.update_states(data,obslike);
                logptilde(i,:) =  cell2mat(self.HMMs{i}.logZ); 
            end
            % add prior
            logptilde = bsxfun(@plus,logptilde,self.pi.loggeomean);
            
            % normalize
            self.p = bsxfun(@minus,logptilde,max(logptilde));            
            self.p = exp(self.p);
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            
% Compute Lower bound
            self.L = -self.KLqprior;
            idx=find(logptilde(:)>-Inf);
            self.L = self.L + sum(self.p(idx).*logptilde(idx));
            idx = find(self.p(:)>0);
            self.L = self.L - sum(self.p(idx).*log(self.p(idx)));
            L = self.L;

            for j=1:modeliters
                if(j>1)
                    obslike = self.obsloglike(data);
                end
                for i=1:self.NC
                    if(j>1)
                        self.HMMs{i}.update_states(data,obslike);
                    end
                    self.HMMs{i}.updateMarkovparms(data,self.p(i,:)); 
                end
                clear pcat
                for i=1:numel(data)
                    pcat{i}=zeros(self.dim,size(data{i},2));
                    for k=1:self.NC
                        pcat{i}=pcat{i}+self.HMMs{k}.p{i}*self.p(k,i);
                    end
                end
                pcat = [pcat{1:end}];
                for k=1:self.dim
                    self.obsModels{k}.update(datacat',pcat(k,:)');  
                end
                
            end
            self.NA = sum(self.p,2);
            self.pi.update(self.NA);

        end
        
        function res = KLqprior(self)
            res = + self.pi.KLqprior;
            for i=1:self.NC
                res = res + self.HMMs{i}.A.KLqprior + self.HMMs{i}.pi0.KLqprior;
            end
            for i=1:self.dim
                res = res + self.obsModels{i}.KLqprior;
            end
        end
        
        function initialize(self,data,iters)

            fprintf('Initializing observation model...')
            
            datacat = [data{:}];
            idx=[];
            for i=1:length(self.obsTypes);
                idx=[idx,self.obsTypes{i}.idx];
            end
            z=kmeans(datacat(idx,:)',self.dim);
            for i=1:self.dim
                self.obsModels{i}.update(datacat(:,z==i)',100*ones(sum(z==i),1)/(1+sum(z==i)));
            end
            
            obslike = self.obsloglike(data);
            
            % fake spectral clustering
            for i=1:length(data)
                [m,idx]=max(obslike{i});
                for k=1:self.dim
                for l=1:self.dim
                    A(k,l) = sum(idx(1:end-1)==k & idx(2:end)==l);
                end
                end
                Avec(i,:)=A(:)';
            end
            
            z = kmeans(Avec,self.NC);

            fprintf('done.\nInitializing HMMs...')
            
%            idx=randperm(length(data),self.NC);

            for i=1:self.NC
            for j=1:iters   
                self.HMMs{i}.update_states(data(z==i),obslike);
                self.HMMs{i}.updateMarkovparms(data(z==i),ones(1,sum(z==i)));
            end
            end
                        
            fprintf('done.\n')
            
            fprintf('Running initial parameter update...')
            self.update(data,1);
            
            fprintf('done.\nPruning unused clusters...')
            self.prune;
            
            fprintf('done.\nRunning second parameter update...')
            self.update(data,1);
            
            fprintf('done.\n')
            self.fillunused(data,20);
            
            fprintf('done.')
        end
        
        function prune(self)
            idx=find(self.NA>0.5);
            if(length(idx)==self.NC)
                return
            end                
            self.NC = length(idx)+1;
            for i=1:length(idx)
                HMMs(i)=self.HMMs(idx(i));
            end
            idx=find(self.NA<0.5);
            HMMs(i+1)=self.HMMs(idx(1));
            
            self.pi=dists.expfam.dirichlet(self.NC,self.pi.alpha_0(1:self.NC,1));
            self.HMMs=HMMs;
            
        end
        
        function fillunused(self,data,iters,neff)            
            if(~exist('neff','var'))
                neff=1;
            end
            % can only be run after update
            idx=find(sum(self.p')<1);
            if(isempty(idx))
                fprintf('No unused clusters. \n')
                return
            end
            fprintf(['Filling ',int2str(length(idx)),' unused clusters\n'])
            for i=1:self.NC
                fitq(i,:) = cell2mat(self.HMMs{i}.logZ);
            end
            m=max(fitq);
            [m,didx]=sort(-m);
            datatemp = data(didx(1:length(idx)));
            for i=1:length(idx)
                for j=1:iters
                    self.HMMs{idx(i)}.update_states(datatemp(i));
                    self.HMMs{idx(i)}.updateMarkovparms(datatemp(i),neff);
                end
            end
            self.pi.alpha=self.pi.alpha_0;
        end
        
        function split(self,data,iters)
            % can only be run after update
            % Choose a cluster to split based upon size.
            idx1 = util.discretesample(  self.pi.mean,1);
            % find 2 smallest cluster to replace
            NA = sum(self.p');
            [m,idx2]=sort(NA);
            idx2=idx2(1);
            
            % Find datapoints assigned to that cluster
            
            [m,pidx]=max(self.p);
            pidx = find(pidx==idx1);
            
            if(length(pidx)<2)
                return
            end
            % Cluster empirical state distributions
            %
            temp=zeros(self.dim,length(pidx));
            for i=1:length(pidx)
                temp(:,i) = sum(self.HMMs{idx1}.p{pidx(i)},2);
            end
            z=kmeans(temp',2);
            
            datatemp1 = data(pidx(z==1));
            datatemp2 = data(pidx(z==2));
            
            self.HMMs{idx1}.A.alpha = self.HMMs{idx1}.A.alpha_0;
            self.HMMs{idx1}.pi0.alpha = self.HMMs{idx1}.pi0.alpha_0;
            
            for j=1:iters
                self.HMMs{idx1}.update_states(datatemp1);
                self.HMMs{idx1}.updateMarkovparms(datatemp1);
                self.HMMs{idx2}.update_states(datatemp2);
                self.HMMs{idx2}.updateMarkovparms(datatemp2);
            end
            
%            self.pi.alpha(idx2)=self.pi.alpha(idx1)/2;
%            self.pi.alpha(idx1)=self.pi.alpha(idx1)/2;
            self.pi.alpha=self.pi.alpha_0;

        end
        
        function plotclusters(self,data,fignum)
            [m,idx]=max(self.p);
            cc=jet(self.dim);

            for j=1:self.NC
                d1=data(idx==j);
                p1=self.HMMs{j}.p(idx==j);
                if(length(d1)>25)
                    tempidx = randi(length(d1),50,1);
                    d1=d1(tempidx);
                    p1=p1(tempidx);
                end
    
                figure(fignum)
                px = ceil(sqrt(self.NC));
                py = ceil(self.NC/px);
                subplot(px,py,j)
                hold on
                if(length(d1)>0)
                    for i=1:length(p1)
                        [m,idx2]=max(p1{i});
                        scatter(d1{i}(2,:),d1{i}(3,:),3*ones(size(d1{i}(3,:))),cc(idx2,:))
                        scatter(2-d1{i}(2,:),d1{i}(1,:),3*ones(size(d1{i}(3,:))),cc(idx2,:))
                    end    
                end
                hold off
            end
            drawnow
        end 
    end
end