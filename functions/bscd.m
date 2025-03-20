function [out]=bscd(sig,okno)
%function [out]=bscd(sig,okno)
out(length(sig))=0;
p2(length(sig))=0;
jm(length(sig))=0;
cit(length(sig))=0;
evid(length(sig))=0;
E6(length(sig))=0;
E5(length(sig))=0;
E4(length(sig))=0;
E3(length(sig))=0;
E2(length(sig))=0;
E1(length(sig))=0;

M1=1;M2=1;
sig=sig(:);
sig=sig/max(abs(sig));
% inicializace %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=fix(okno/2);
%m=stred;


data=sig(1:okno);
N=length(data);
    for j=1:N
        if j <= m
                   G(j,:)=[1 0];
        end;        
        if j > m
                   G(j,:)=[0 1];        
        end;
    end;

    D=data'*data;
    CHI=data'*G;
    GTG=G'*G;
    FI=inv(GTG);
    DELTA=det(GTG);
    
    cit=((-N+M1+M2)/2)*log10(D-CHI*FI*CHI');
    jm=0.5*log10(DELTA);
%    p2(m)=cit-jm;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vypocet bayesovske evidence
% vytvoreni matice G_E (pro bayesovskou evidenci)
%%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j=1:N
                G_E(j,:)=[1];
    end;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
CHI_E  = data'*G_E;
GTG_E  = G_E'*G_E;
FI_E   = inv(GTG_E);
DELTA_E= det(GTG_E);

B=FI_E*CHI_E';
BTB=B'*B;
FTF=CHI_E*FI_E*CHI_E';
%evid(m)=((M1-N)/2)*log10(pi)+log10(gamma((N-M1)/2))-0.5*log10(det(G_E'*G_E))-((N-M1)/2)*log10(D-FTF);
%evid(m)=((-N)/2)*log10(pi)-0.5*log10(det(G_E'*G_E))+log10(gamma((M1)/2))+log10(gamma((N-M1)/2))-((N-M1)/2)*log10(D-FTF)+0.5*M1*log10(BTB);
E1=((-N)/2)*log10(pi);
E2=-0.5*log10(DELTA_E);
E3=+log10(gamma((M1)/2));
E4=+log10(gamma((N-M1)/2));
E5=-((N-M1)/2)*log10(D-FTF);
E6=-0.5*M1*log10(BTB);
evid(m)=E1+E2+E3+E5+E6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p2(m)=cit-jm-evid(m);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for mm=m+1:length(sig)-m
% pridani novych dat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d2=sig(mm+m);
    G2=[0 1];
    D=D+d2'*d2;
    CHI=CHI+d2'*G2;
    W=FI*G2';
    LAMBDA=(1+G2*W);
    DELTA=DELTA*det(LAMBDA);
    FI=FI-W*inv(LAMBDA)*W';
    % prepocet BE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G2_E=[1];
    CHI_E=CHI_E+d2'*G2_E;
    W_E=FI_E*G2_E';
    LAMBDA_E=(1+G2_E*W_E);
    DELTA_E=DELTA_E*det(LAMBDA_E);
    FI_E=FI_E-W_E*inv(LAMBDA_E)*W_E';    
% vlozeni nul %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D=D-sig(mm-m)'*sig(mm-m);
    Z=[1 0];
    CHI=CHI - sig(mm-m)*Z;
    W=FI*Z';
    LAMBDA=(1-Z*W);         
    DELTA=LAMBDA*DELTA; 
    FI=FI+(1/LAMBDA)*W*W';         
    % prepocet BE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Z_E=[1];
    CHI_E=CHI_E - sig(mm-m)*Z_E;
    W_E=FI_E*Z_E';
    LAMBDA_E=(1-Z_E*W_E);         
    DELTA_E=LAMBDA_E*DELTA_E; 
    FI_E=FI_E+(1/LAMBDA_E)*W_E*W_E';         

B=FI_E*CHI_E';
BTB=B'*B;
FTF=CHI_E*FI_E*CHI_E';
%evid(mm)=((M1-N)/2)*log10(pi)+log10(gamma((N-M1)/2))-0.5*log10(det(G_E'*G_E))-((N-M1)/2)*log10(D-FTF);
%evid(mm)=((-N)/2)*log10(pi)-0.5*log10(det(G_E'*G_E))+log10(gamma((M1)/2))+log10(gamma((N-M1)/2))-((N-M1)/2)*log10(D-FTF)+0.5*M1*log10(BTB);
E1=((-N)/2)*log10(pi);
E2=-0.5*log10(DELTA_E);
E3=+log10(gamma((M1)/2));
E4=+log10(gamma((N-M1)/2));
E5=-((N-M1)/2)*log10(D-FTF);
E6=+0.5*M1*log10(BTB);
evid(mm)=E1+E2+E3+E5-E6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% posunuti pozice m+1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R=[0 1];
    CHI=CHI - sig(mm)*R;
    W=FI*R';
    LAMBDA=(1-R*W);         
    DELTA=LAMBDA*DELTA; 
    FI=FI+(1/LAMBDA)*W*W';         
    Q=[1 0];
    CHI=CHI + sig(mm)*Q;
    W=FI*Q';
    LAMBDA=(1+Q*W);         
    DELTA=LAMBDA*DELTA;
    FI=FI-(1/LAMBDA)*W*W';   
    
    cit=((-N+M1+M2)/2)*log10(D-CHI*FI*CHI');
    jm=0.5*log10(DELTA);
    p2(mm)=real(cit)-real(jm)-real(evid(mm));
end;
x=find(p2); minp=min(p2(x)); out=p2-minp; 
out=out.*(out > 0);