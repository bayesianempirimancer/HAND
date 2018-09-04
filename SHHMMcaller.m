% SHHMM CALLER
% 
NC=8;
NS=10;
D=3;
model = SHHMM(NC,NS,D,[],ones(NC,1),ones(NS,NS),ones(NS,1));
tic;
Niters = 25;
for i=1:Niters
    L(i) = model.update(databits);
    fprintf(['Completed ',num2str(i),' iterations in ',num2str(toc),' seconds\n']);
    fprintf(['   with a ELBO of ',num2str(L(i)),'\n']);
end

