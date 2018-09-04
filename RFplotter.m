function F = RFplotter(temp,Y,X,V,D)

Nx=13;
Ny=13;
Nt=13;
Nsf=29;

W=Y'*X/length(Y);  %classic reverse correlation;
W30=W*V'*diag(sqrt(D));
Wf=W30(1,end-Nsf+1:end);
W30 = W30(1,1:end-Nsf);
W30=reshape(W30,[Nx*Ny,Nt]);
W30=reshape(W30,[Nx,Ny,Nt]);

X=bsxfun(@times,X,sqrt(D'))*V';

temp=temp*diag(sqrt(D))*V';

spikefilter1 = temp(1,end-Nsf+1:end);
if(size(temp,1)==1)
    spikefilter2=spikefilter1;
else
    spikefilter2 = temp(2,end-Nsf+1:end);
end
figure(4)
plot(spikefilter1,'b'), hold on; plot(spikefilter2,'r'); plot(Wf,'k'); hold off;
temp=temp(:,1:end-Nsf);

spikefilter1contrib = var(X(:,end-Nsf+1:end)*spikefilter1')
SpatioTemporalFiltercontribs = var(X(:,1:end-Nsf)*temp')
spikefilter2contrib = var(X(:,end-Nsf+1:end)*spikefilter2')

W31=reshape(temp(1,:),[Nx*Ny,Nt]);
figure(3)
[V1,D1]=eig(cov(W31));
W31rank1=W31*V1(:,end);
subplot(2,2,1), plot(V1*sqrt(D1))
subplot(2,2,3), imagesc(reshape(W31rank1,[Nx,Ny])),colorbar
if(size(temp,1)==1)
    W32=W31;
else
W32=reshape(temp(2,:),[Nx*Ny,Nt]);
[V2,D2]=eig(cov(W32));
W32rank1=W32*V2(:,end);
subplot(2,2,2), plot(V2*sqrt(D2))
subplot(2,2,4), imagesc(reshape(W32rank1,[Nx,Ny])),colorbar
end

SpatialFilterCorr = corr(W31rank1,W32rank1)
TemporalFilterCorr = corr(V1(:,end),V2(:,end))
TemporalFilterCrossCorr = crosscorr(V1(:,end),V2(:,end));
TemporalFilterAutoCorr1 = crosscorr(V1(:,end),V1(:,end));
TemporalFilterAutoCorr2 = crosscorr(V2(:,end),V2(:,end));
figure(2)
plot(TemporalFilterCrossCorr,'k')
hold on
plot(TemporalFilterAutoCorr1,'b')
plot(TemporalFilterAutoCorr2,'r')
hold off
W31=reshape(W31,[Nx,Ny,Nt]);
W32=reshape(W32,[Nx,Ny,Nt]);

%B=exp(-4*(-Nx+1:Nx-1).^2/2)'*exp(-4*(-Ny+1:Ny-1).^2/2);

B=exp(-100*(-Nx+1:Nx-1).^2/2)'*exp(-100*(-Ny+1:Ny-1).^2/2);
B=B'*B;
for i=1:Nt
    W30(:,:,i)=conv2(W30(:,:,i),B,'same');
    W31(:,:,i)=conv2(W31(:,:,i),B,'same');
    W32(:,:,i)=conv2(W32(:,:,i),B,'same');
end
    

c0min=min(W30(:));
c0max=max(W30(:));
c0fmin=Inf;
c0fmax=-Inf;
c1min=min(W31(:));
c1max=max(W31(:));
c1fmin=Inf;
c1fmax=-Inf;
c2min=min(W32(:));
c2max=max(W32(:));
c2fmin=Inf;
c2fmax=-Inf;

% c0min=Inf;
% c0max=-Inf;
% c0fmin=Inf;
% c0fmax=-Inf;
% c1min=Inf;
% c1max=-Inf;
% c1fmin=Inf;
% c1fmax=-Inf;
% c2min=Inf;
% c2max=-Inf;
% c2fmin=Inf;
% c2fmax=-Inf;


for i=1:Nt
%    c0min=min(min(min(W30(:,:,i))),c0min);
%    c0max=max(max(min(W30(:,:,i))),c0max);
    c0fmin=min(min(min(abs(fft2(W30(:,:,i))).^2)),c0fmin);
    c0fmax=max(max(max(abs(fft2(W30(:,:,i))).^2)),c0fmax);
    
%    c1min=min(min(min(W31(:,:,i))),c1min);
%    c1max=max(max(min(W31(:,:,i))),c1max);
    c1fmin=min(min(min(abs(fft2(W31(:,:,i))).^2)),c1fmin);
    c1fmax=max(max(max(abs(fft2(W31(:,:,i))).^2)),c1fmax);
    mu1(i)=mean(mean(W31(:,:,i)));
    
%    c2min=min(min(min(W32(:,:,i))),c2min);
%    c2max=max(max(min(W32(:,:,i))),c2max);
    c2fmin=min(min(min(abs(fft2(W32(:,:,i))).^2)),c2fmin);
    c2fmax=max(max(max(abs(fft2(W32(:,:,i))).^2)),c2fmax);
    mu2(i)=mean(mean(W32(:,:,i)));

end



for i=12:-1:1
figure(5)
subplot(3,2,1), imagesc(W30(:,:,i)), caxis([c0min,c0max]), colorbar
title(['t = - ',num2str(i)])
temp=abs(fft2(W30(:,:,i))).^2;
temp=temp(1:Nx,1:Ny);
%temp(1,1)=0;
subplot(3,2,2), imagesc(temp), caxis([c0fmin,c0fmax]), colorbar
subplot(3,2,3), imagesc(W31(:,:,i)), caxis([c1min,c1max]), colorbar
temp=abs(fft2(W31(:,:,i))).^2;
temp=temp(1:Nx,1:Ny);
%temp(1,1)=0;
subplot(3,2,4), imagesc(temp), caxis([c1fmin,c1fmax]), colorbar
subplot(3,2,5), imagesc(W32(:,:,i)), caxis([c2min,c2max]), colorbar
temp=abs(fft2(W32(:,:,i))).^2;
temp=temp(1:Nx,1:Ny);
%temp(1,1)=0;
subplot(3,2,6), imagesc(temp), caxis([c2fmin,c2fmax]), colorbar
drawnow
F(12-i+1)=getframe(5);
pause
end

figure(5)
clf
axis off
movie(F,5,2);
end