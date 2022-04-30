function [B, B1, B2] = RWSH12(X1, X2, Y, param, XTest, YTest)
% WASH solves the following problem
% beta*|E|_1 + gamma*sum_{i=0}^n eta_i*|Xi-Wi*V| + mu*|PiXi-SG|
% + phi*|B-V| + ksai*|V'B-rD|
% Y = SG + E, X0=G, VV' = nI, V1=0


X1 = X1'; X2=X2'; Y = Y';
[dx1, n ] = size(X1); [dx2, ~ ] = size(X2); [c, ~] = size(Y);
beta = param.beta;
gamma = param.gamma;
mu = param.mu;
lambda = param.lambda;
eta0 = param.eta0;
eta1 = param.eta1;
eta2 = param.eta2;
r = param.nbits;
ksai = 1;
phi = param.phi;

%rand('seed',2021);
sel_sample = X1(:,randsample(n, 10000),:);
[pcaW, ~] = eigs(cov(sel_sample'), r);
V1_c = pcaW'*X1;
sel_sample = X2(:,randsample(n, 10000),:);
[pcaW, ~] = eigs(cov(sel_sample'), r);
V2_c = pcaW'*X2;
V =  (V1_c+V2_c)/2  ;
k = round(c);
sel_sample = Y(:,randsample(n, 5000),:);
[pcaW, ~] = eigs(cov(sel_sample'), k);
G = pcaW'*Y;

E = zeros(size(Y));
B = sign(V); B(B==0)=-1;

X1X1 = X1*X1';
X2X2 = X2*X2'; 
P1 = (mu*(Y-E)*X1')*pinv(mu*X1X1 + lambda*eye(dx1));
P2 = (mu*(Y-E)*X2')*pinv(mu*X2X2 + lambda*eye(dx2));
F1 = zeros(size(Y));
Yn = Y./repmat(sqrt(sum(Y.*Y))+1e-8,[size(Y, 1),1]);
rho = 1;
X1in = pinv(mu*X1X1 + lambda*eye(dx1));
X2in = pinv(mu*X2X2 + lambda*eye(dx2));

for iter = 1:param.iter  
    % B
     B = sign(phi*V + r*ksai*V*Yn'*Yn);
     B(B==0)=-1;
     
         %W
     W0 = gamma*eta0*G*V'/(gamma*eta0*V*V'+lambda*eye(r));
     W1 = gamma*eta1*X1*V'/(gamma*eta1*V*V'+lambda*eye(r));
     W2 = gamma*eta2*X2*V'/(gamma*eta2*V*V'+lambda*eye(r));

    % 
   S = (rho/2*(Y-E+1/rho*F1)*G'+mu*(P1*X1+P2*X2)*G')/((rho/2 + 2*mu)*G*G'+1e-8*eye(size(G,1)));
   
   %G
   G = ((rho/2+2*mu)*S'*S + (gamma*eta0+ 1e-8)*eye(k))\(rho/2*S'*(Y-E+1/rho*F1) + gamma*eta0*W0*V + mu*S'*(P1*X1+P2*X2) );
   
   Yn = Y -E;
   Yn = Yn./repmat(sqrt(sum(Yn.*Yn)),[size(Yn, 1),1]);
   
     % P
      P1 = (mu*S*G*X1')*X1in ;
      P2 = (mu*S*G*X2')*X2in ;
    
   % E
     Etp = Y - S*G + 1/rho*F1;
     E = sign(Etp).*max(abs(Etp)- beta/rho,0); 
   
     % V       
     Z = gamma*(eta0*W0'*G+eta1*W1'*X1+eta2*W2'*X2) + r*ksai*B*Yn'*Yn + phi*B;
     Z = Z' ;
     Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-6);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,r-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[Q Q_]';
     V = V';       
     
    F1 = F1 + rho*(Y - S*G -E);
    rho = min(1e4, 1.3*rho);
end

    G1 = (B*X1')/(X1X1+1e-3*eye(size(X1,1)));
    G2 = (B*X2')/(X2X2+1e-3*eye(size(X2,1)));
    B = B'>0;
    B1 = XTest*G1'>0;
    B2 = YTest*G2'>0;

end

