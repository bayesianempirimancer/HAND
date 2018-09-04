function [ res, P ] = hmmdist( m1,m2 )

% Adhoc distance measure between hidden markov models
% Step 1:  sort states using observation models
% Step 2:  permute transition probability matrices
% res(1) = sum square distance between state means
% res(2) = sum square distance between transition probabilities

idx=[];
for i=1:length(m1.obsTypes)
    idx=[idx,m1.obsTypes{i}.idx];
end
for i=1:m1.dim
    mu1(i,:)=m1.obsModels{i}.mean;
    mu2(i,:)=m2.obsModels{i}.mean;
end
mu1=mu1(:,idx);
mu2=mu2(:,idx);

for i=1:m1.dim
for j=1:m2.dim
    d(i,j)=sum((mu1(i,:)-mu2(j,:)).^2);
end
end
dsave=d;

for k=1:m1.dim
    [m,idxi]=min(d);
    [msave(k),j]=min(m);

    i=idxi(j);

    P(i,j)=1;
    d(i,:)=Inf;
    d(:,j)=Inf;
end

% d=d*P';
% mu2=P*mu2;

res(1) = sum(msave);
res(2) = sum(sum((m1.A.mean - P*m2.A.mean*P').^2));

figure(1), imagesc(m1.A.mean), colorbar
figure(2), imagesc(P*m2.A.mean*P'), colorbar


end

