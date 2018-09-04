%GMMcaller
% X is data
% Y is externally generated labels so make edits here:
% So you can do this:
NC=10;  %max number of clusters
dim=3; % dimensionality of observation space (number of indices)


X=[dSPNs_high;dSPNs_low;];
Y=ones(size(X,1),1);
Y(size(dSPNs_high,1)+1:size(dSPNs_high,1)+size(dSPNs_low,1),1)=0;

% %% OR this
% X=[iSPNs_high;iSPNs_low;];
% Y=ones(size(X,1),1);
% Y(size(iSPNs_high,1)+1:size(iSPNs_high,1)+size(iSPNs_low,1),1)=0;


% A bit of useful pre-processing
% 
X(:,2)=X(:,2)-X(:,1);
mu=mean(X);
D=var(X);
V=eye(size(X,2));

X=bsxfun(@plus,X,-mu);
X=X*diag(1./sqrt(D));

GMM=GaussianMixtureModel(NC,dim,1);
GMM.update(X,1000);


[m,idx]=max(GMM.NA);
flip=0;
passign_high = GMM.p(:,idx);
if(corr(passign_high,Y)<0)
    passign_high=1-passign_high;
    flip=1;
end

%brute force to find best decision criterion
d=[0:0.01:1];
for k=1:length(d)
    cd(k)=sum(passign_high>d(k) & Y==1)/sum(Y==1);
    fp(k)=sum(passign_high>d(k) & Y==0)/sum(Y==0);
    pc(k)=sum(passign_high>d(k) & Y==1) + sum(passign_high<=d(k) & Y==0);
end
pc=pc/size(X,1);
[m,idxdc]=max(pc);
opt_dc = d(idxdc);
figure(2)
plot(fp,cd,d,pc)
legend('ROC','PC')
title('ROC / Percent Correct')
xlabel('False Positive Rate / Decision criterion')
%alternatively use prior
prior = GMM.pi.mean;
if(flip==0)
    opt_dc = prior(idx)
else
    opt_dc = 1-prior(idx)
end
Cpi=GMM.pi.variance();
opt_dc_standard_deviation = sqrt(Cpi(idx))
    
percent_correct = GMM.plotclusters(X,1,Y,opt_dc,V,D,mu)


