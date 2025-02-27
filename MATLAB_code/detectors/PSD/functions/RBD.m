function [out]=rbd(signal,window_length,AR_order_left,AR_order_right,Bayesian_Evidence_order)
%
% Recursive Bayesian (Autoregressive Changepoint) Detector
% Usage: [out] = bacdn(signal, window_length, AR_order_left, AR_order_right, Bayesian_Evidence_order)
%
% Input:  signal ............ 1-D data vector
%         window_length ..... length of sliding window
%                             (default 400), min~200, max~5000
%         AR_order .......... order of AR model
%                             (default 6)
% Output: out ................~(d_cep1).^2
%                             is proportional to the signal changes 
%                             
% by Roman Cmejla, cmejla@fel.cvut.cz
% 2006Aug15, ver.1.00  

if (nargin == 0)
			disp('Usage: [out] = bacdn(signal, window_length, AR_order)');
			return;
end
if (nargin == 1)
    AR_order_left=6;
    AR_order_right=6;
    Bayesian_Evidence_order=6;
    window_length=400;
end;
if (nargin == 2)
    AR_order_left=6;
    AR_order_right=6;
    Bayesian_Evidence_order=6;
end;
if (nargin == 3)
    AR_order_right=AR_order_left;
    Bayesian_Evidence_order=AR_order_left;
end;
if (nargin == 4)
    Bayesian_Evidence_order=AR_order_left;
end;
p2(length(signal))=0;
jm(length(signal))=0;
cit(length(signal))=0;
evid(length(signal))=0;
E6(length(signal))=0;
E5(length(signal))=0;
E4(length(signal))=0;
E3(length(signal))=0;
E2(length(signal))=0;
E1(length(signal))=0;

sig=signal;
M1=AR_order_left;
M2=AR_order_right;
ME=Bayesian_Evidence_order;

okno=window_length;

p2=zeros(1,length(sig));
sig=sig(:);
sig=sig/max(abs(sig));
% inicializace %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=fix(okno/2);

%data=sig(1:okno).*hamming(length(sig(1:okno)));
data=sig(1:okno);

G(length(data-m),M1+M2)=0;
N=length(data);
    for j=2:M1
        if j <= m
                   G(j,:)=[data(j-1:-1:1)' zeros(1,M1-j+1) zeros(1,M2)];
        end;        
        if j > m
                   G(j,:)=[zeros(1,M1) data(j-1:-1:1)' zeros(1,M2-j+1)];        
        end;
    end;
    for j=(M1+1):N
            if j <= m
                G(j,:)=[data(j-1:-1:j-M1)' zeros(1,M2)];
            end;
            if j > m
                G(j,:)=[zeros(1,M1) data(j-1:-1:j-M2)'];                
            end;
    end;

    D=data'*data;
    CHI=data'*G;
    GTG=G'*G;
    FI=inv(GTG);
    DELTA=det(GTG);
    cit(m)=log(D-CHI*FI*CHI'); %!!!!!!!!!!!!!!!!!!!
%    cit(m)=((-N+M1)/2)*log(D-CHI*FI*CHI');  
    jm(m)=0.5*log(DELTA);
%    p2(m)=cit-jm;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vypocet bayesovske evidence
% vytvoreni matice G_E (pro bayesovskou evidenci)
%%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%
%G_E(N,ME)=0;
    for j=2:ME
                G_E(j,:)=[data(j-1:-1:1)' zeros(1,ME-j+1)];
    end;
    for j=(ME+1):N
                G_E(j,:)=[data(j-1:-1:j-ME)'];
    end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
CHI_E  = data'*G_E;
GTG_E  = G_E'*G_E;
FI_E   = inv(GTG_E);
DELTA_E= det(GTG_E);

B=FI_E*CHI_E';
BTB=B'*B;
FTF=CHI_E*FI_E*CHI_E';
%evid(m)=((ME-N)/2)*log(pi)+log(gamma((N-ME)/2))-0.5*log(det(G_E'*G_E))-((N-ME)/2)*log(D-FTF);
%evid(m)=((-N)/2)*log(pi)-0.5*log(det(G_E'*G_E))+log(gamma((ME)/2))+log(gamma((N-ME)/2))-((N-ME)/2)*log(D-FTF)+0.5*ME*log(BTB);
E1(m)=((-N)/2)*log(pi);
E2(m)=-0.5*log(DELTA_E);
E3(m)=+log(gamma((ME)/2));
E4(m)=+log(gamma((N-ME)/2));
% E5(m)=-((N-ME)/2)*log(D-FTF); 
E5(m)=log(D-FTF); % !!!!!!!!!!!!!!!!!!!!!!!!!!
E6(m)=-0.5*ME*log(BTB);
evid(m)=E1(m)+E2(m)+E3(m)+E5(m)+E6(m);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p2(m)=cit(m)-jm(m)-evid(m);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for mm=m+1:length(sig)-m
% pridani novych dat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d2=sig(mm+m);
    G2=[zeros(1,M1) sig(mm+m-1:-1:mm+m-M2)'];
    D=D+d2'*d2;
    CHI=CHI+d2'*G2;
    W=FI*G2';
    LAMBDA=(1+G2*W);
    DELTA=DELTA*det(LAMBDA);
    FI=FI-W*inv(LAMBDA)*W';
    % prepocet BE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G2_E=[sig(mm+m-1:-1:mm+m-ME)'];
    CHI_E=CHI_E+d2'*G2_E;
    W_E=FI_E*G2_E';
    LAMBDA_E=(1+G2_E*W_E);
    DELTA_E=DELTA_E*det(LAMBDA_E);
    FI_E=FI_E-W_E*inv(LAMBDA_E)*W_E';    
% vlozeni nul %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D=D-sig(mm-m)'*sig(mm-m);
    if mm-m > M1
        Z=[sig(mm-1-m:-1:mm-M1-m)' zeros(1,M2)];
    else
        Z=[sig(mm-1-m:-1:1)' zeros(1,M2) zeros(1,M1-mm+m+1)];
    end
    CHI=CHI - sig(mm-m)*Z;
    W=FI*Z';
    LAMBDA=(1-Z*W);         
    DELTA=LAMBDA*DELTA; 
    FI=FI+(1/LAMBDA)*W*W';         
    % prepocet BE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if mm-m > ME
        Z_E=[sig(mm-1-m:-1:mm-ME-m)'];
    else
        Z_E=[sig(mm-1-m:-1:1)' zeros(1,ME-mm+m+1)];
    end
    CHI_E=CHI_E - sig(mm-m)*Z_E;
    W_E=FI_E*Z_E';
    LAMBDA_E=(1-Z_E*W_E);         
    DELTA_E=LAMBDA_E*DELTA_E; 
    FI_E=FI_E+(1/LAMBDA_E)*W_E*W_E';         

B=FI_E*CHI_E';
BTB=B'*B;
FTF=CHI_E*FI_E*CHI_E';
%evid(mm)=((ME-N)/2)*log(pi)+log(gamma((N-ME)/2))-0.5*log(det(G_E'*G_E))-((N-ME)/2)*log(D-FTF);
%evid(mm)=((-N)/2)*log(pi)-0.5*log(det(G_E'*G_E))+log(gamma((ME)/2))+log(gamma((N-ME)/2))-((N-ME)/2)*log(D-FTF)+0.5*ME*log(BTB);
E1(mm)=((-N)/2)*log(pi);
E2(mm)=-0.5*log(DELTA_E);
E3(mm)=+log(gamma((ME)/2));
E4(mm)=+log(gamma((N-ME)/2));
% E5(mm)=-((N-ME)/2)*log(D-FTF);
E5(mm)=log(D-FTF); % !!!!!!!!!!!!!
E6(mm)=+0.5*ME*log(BTB);

evid(mm)=E1(mm)+E2(mm)+E3(mm)+E5(mm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% posunuti pozice m+1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R=[zeros(1,M1) sig(mm-1:-1:mm-M2)'];
    CHI=CHI - sig(mm)*R;
    W=FI*R';
    LAMBDA=(1-R*W);         
    DELTA=LAMBDA*DELTA; 
    FI=FI+(1/LAMBDA)*W*W';         
    Q=[sig(mm-1:-1:mm-M1)' zeros(1,M2)];
    CHI=CHI + sig(mm)*Q;
    W=FI*Q';
    LAMBDA=(1+Q*W);         
    DELTA=LAMBDA*DELTA;
    FI=FI-(1/LAMBDA)*W*W';   
    cit(mm)=log(D-CHI*FI*CHI'); % !!!!!!!!!!!!!!!!!!
%    cit(mm)=((-N+M1+M2)/2)*log(D-CHI*FI*CHI');
    jm(mm)=0.5*log(DELTA);
    p2(mm)=real(cit(mm))-real(jm(mm))-real(evid(mm));
end;
CSF=real(E5)-real(cit);
out=4*CSF(round(okno/2)+1:end);
out=[zeros(1,m) out];