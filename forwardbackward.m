function [p,SExx,logZ] = forwardbackward(loglike,logA,logpi)  
    % forward propogation
    [dim,T]=size(loglike);
    
    A=exp(logA);
    pi=exp(logpi);
%    [loglike,logz] = lognormalize(loglike,1);
    
    logz=max(loglike);
    loglike=bsxfun(@minus,loglike,logz);
    
    like = exp(loglike);
    z = exp( logz );
    a = zeros(dim,T);

    a(:,1) = like(:,1).*pi;            
%            [a(:,1),z(1)] = util.normalize(a(:,1),z(1));
    a_sum = sum(a(:,1),1);
    a(:,1)=a(:,1)/a_sum;
    z(1) = z(1)*a_sum;

    for t = 2:T
        a(:,t) = like(:,t) .* (A'*a(:,t-1));
%                [a(:,t),z(t)] = util.normalize(a(:,t),z(t));
        a_sum = sum(a(:,t),1);
        a(:,t)=a(:,t)/a_sum;
        z(t) = z(t)*a_sum;
    end

    % backward propogation
    b = zeros(dim,T);
    b(:,T) = 1/dim;
    for t = T:-1:2
        b(:,t-1) = A * (like(:,t) .* b(:,t));
        b(:,t-1) = b(:,t-1) / sum(b(:,t-1));
    end

    p = a .* b;
    p = bsxfun(@rdivide,p,sum(p,1));

    xi=zeros(dim,dim,T-1);
       
    for i=1:dim
    for j=1:dim
        xi(i,j,:)=(a(i,1:end-1).*b(j,2:end).*like(j,2:end))*A(i,j);
    end
    end
%     xi2 = bsxfun(@times,permute(a(:,1:end-1),[1,3,2]), ...
%         permute(b(:,2:end).*like(:,2:end),[3,1,2]));   
%     xi2 = bsxfun(@times,xi2,A);

    
    xi = bsxfun(@rdivide,xi,sum(sum(xi,1),2));
    xi(isnan(xi(:)))=0;
    SExx = squeeze(sum(xi,3));
    logZ = sum(log(z));
end

% function [s,z] = lognormalize(A,dim)
% %LOGNORMALIZE: normalize the matrix A along dimension dim
% 
% % find log of normalizing constant
% if dim == 1
%     z = logsumexp(A')';
% else
%     z = logsumexp(A);
% end
% 
% % normalize by subtracting off constant
% s = bsxfun(@minus,A,z);
% 
% end
% 
% function s = logsumexp(a)
% if size(a,2) < 2
%     s = a;
% else
%     y = max(a,[],2);
%     s = bsxfun(@minus,a,y);
%     s = y + log(sum(exp(s),2));
%     %s(~isfinite(y)) = y(~isfinite(y));
% end
% end
