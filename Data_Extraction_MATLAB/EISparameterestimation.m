function main
global cons;
global bat
mydatadeal;
myoriginalplot;
mydealplot;
parameters_primary_estimation;
cons.Freq=cons.dataremain(:,1);
cons.ReZ=cons.dataremain(:,2);
cons.NImZ=cons.dataremain(:,3);

for i=1:1:90
cons.Freq1(i)=cons.Freq(i+5,1);
cons.ReZ1(i)=cons.ReZ(i+5,1);
cons.NImZ1(i)=cons.NImZ(i+5,1);
end


X(1)=bat.R0;
X(2)=bat.R1;
X(3)=bat.C1;
X(4)=bat.C2;

myEISplotRCCesti(cons.Freq1,X,4);

myEISRCCoptfit(cons.Freq1,cons.ReZ1,cons.NImZ1,X);
myEISplotRCCopti(cons.Freq1,cons.xout,5);

xlswrite('S00418002054_1st Cycle_C01consxout.xlsx',cons.xout);


end

function mydatadeal

global cons;
clear cons.dataremain;
cons.datain=xlsread('S00418002054_1st Cycle_C01.xlsx','A62:C251');
    j=1;
for i=1:1:size(cons.datain,1)

    if cons.datain(i,3) >= 0
        cons.dataremain(j,:)=cons.datain(i,:);   
        j=j+1;
   
    end
    
    
end
xlswrite('S00418002054_1st Cycle_C01dataremain.xlsx',cons.dataremain);
% disp(cons.datain(1,3))
end

function myoriginalplot
global cons;
for i=1:1:size(cons.datain,1)
ReZ(i)=cons.datain(i,2)
NImZ(i)=cons.datain(i,3)     
end
figure(1);
plot(ReZ,NImZ,'*');%'b','lineWidth',4
xlabel('ReZ');
ylabel('-ImZ');
title('S00418002054_1st Cycle_C01originaldata')

hold on;
saveas(figure(1),'S00418002054_1st Cycle_C01originaldata.fig');
close(figure(1));
end



function mydealplot
global cons;

i=1;
while cons.dataremain(i,1)>cons.dataremain(i+1,1)
    
ReZ(i)=cons.dataremain(i,2)
NImZ(i)=cons.dataremain(i,3)  
if i==(size(cons.dataremain,1)-1)
    break;
end
i=i+1;

end
figure(2);

% plot(ReZ,NImZ,'--pr','lineWidth',4,'MarkerEdgeColor','b','MarkerFaceColor','b');
plot(ReZ,NImZ,'*');
xlabel('ReZ');
ylabel('-ImZ');
title('S00418002054_1st Cycle_C01dealdata')
hold on;
saveas(figure(2),'S00418002054_1st Cycle_C01dealdata.fig');
close(figure(2));
end

function parameters_primary_estimation
global cons;
global bat;
v=1;
while cons.dataremain(v,1)>cons.dataremain(v+1,1)
 freq(v)=cons.dataremain(v,1);
ReZ(v)=cons.dataremain(v,2);
NImZ(v)=cons.dataremain(v,3); 
if v==(size(cons.dataremain,1)-1)
    break;
end
 v=v+1;
end

NImZA=smooth(NImZ,20);
% disp(A);
figure(3)
plot(ReZ,NImZ,'*');
hold on;
plot(ReZ,NImZA,'r','lineWidth',2);
hold on;
% plot(ReZ,NImZ,'b','lineWidth',4);
xlabel('ReZ');
ylabel('-ImZ');
title('S00418002054_1st Cycle_C01smoothplot');

saveas(figure(3),'S00418002054_1st Cycle_C01smoothplot.fig');
close(figure(3));
vmax=v-1;
disp(vmax);
k=0;
for i=2:1:vmax-1
    if (NImZA(i)>=NImZA(i-1))&&(NImZA(i)>=NImZA(i+1))
     ReZmidH= ReZ(i); %circle highest point
     freqmidH=freq(i);
     RmidHnum=i;
     
%      break;
    end
    if (NImZA(i)<=NImZA(i-1))&&(NImZA(i)<=NImZA(i+1))
     ReZmidL= ReZ(i); %outline lowest point
     freqmidL=freq(i);   
     RmidLnum=i;
    end
    
    
end


bat.R0=ReZ(1);
bat.R1=ReZmidL-ReZ(1);
bat.C1=1/(2*pi*freqmidH*bat.R1);
bat.C2=1/(2*pi*freqmidL*NImZA(RmidLnum));

disp(bat.R0);
disp(bat.R1);
disp(bat.C1);
disp(bat.C2);

end

function myEISplotRCCesti(arate,X,p)
global cons;
global bat;
[ReZ,NImZ]=calculateRCC(arate,X)
%   disp(ReZ);
%   disp(NImZ);
disp(size(NImZ));

figure(p);
plot(ReZ,NImZ,'r','lineWidth',2);
% plot(ReZ,NImZ);
hold on;

plot(cons.ReZ1,cons.NImZ1,'*');
hold on;
xlabel('ReZ');
ylabel('-ImZ');
title('S00418002054_1st Cycle_C01estimateresult');
saveas(figure(p),'S00418002054_1st Cycle_C01estimateresult.fig');
close(figure(p));
end

function myEISplotRCCopti(arate,X,p)
global cons;
global bat;
[ReZ,NImZ]=calculateRCC(arate,X)
%   disp(ReZ);
%   disp(NImZ);
disp(size(NImZ));

figure(p);
plot(ReZ,NImZ,'r','lineWidth',2);
% plot(ReZ,NImZ);
hold on;

plot(cons.ReZ1,cons.NImZ1,'*');
hold on;
xlabel('ReZ');
ylabel('-ImZ');
title('S00418002054_1st Cycle_C01optimizationresult');
saveas(figure(p),'S00418002054_1st Cycle_C01optimizationresult.fig');
close(figure(p));

end

function [ReZ,NImZ]=calculateRCC(arate,X)
global cons;
global bat;
pi=3.1415926;
aratesize=size(arate,2);

disp(aratesize);
  for i=1:1:aratesize
      arate(i)=2*pi*arate(i);
ZR0=X(1);
ZR1=X(2);
ZC1=1/(j*arate(i)*X(3));
ZC2=1/(j*arate(i)*X(4));
Z(i)=ZR0+ZR1*ZC1/(ZR1+ZC1);%+ZC2;
 ReZ(i)=real(Z(i));
 NImZ(i)=-imag(Z(i)) ;  
  end
end


function myEISRCCoptfit(arate,ReZ0,NImZ0,X0)
global cons;
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0,0,0,0];
ub=[inf,inf,inf,inf];
% [y,fval,exitfalg]=fmincon(@distance,X0,A,b,Aeq,beq,lb,ub);

[y,fval,exitflag,output]=fminsearchbnd(@distance,X0,lb,ub)

function dist=distance(X)
[ReZ,NImZ]=calculateRCC(arate,X);
s=0;
for i=1:1:size(ReZ,2)
 s=s+((ReZ(i)-ReZ0(i))^2+(NImZ(i)-NImZ0(i))^2)/((ReZ0(i))^2+(NImZ0(i))^2);  
end
dist=s;
end
cons.xout=y;
end


%%GOOD SEARCHING METHOD FOR CONSTRAINTS
function [x,fval,exitflag,output]=fminsearchbnd(fun,x0,LB,UB,options,varargin)
% fminsearchbnd: fminsearch, but with bound constraints by transformation
% usage: fminsearchbnd(fun,x0,LB,UB,options,p1,p2,...)
% 
% arguments:
%  LB - lower bound vector or array, must be the same size as x0
%
%       If no lower bounds exist for one of the variables, then
%       supply -inf for that variable.
%
%       If no lower bounds at all, then LB may be left empty.
%
%  UB - upper bound vector or array, must be the same size as x0
%
%       If no upper bounds exist for one of the variables, then
%       supply +inf for that variable.
%
%       If no upper bounds at all, then UB may be left empty.
%
%  See fminsearch for all other arguments and options.
%  Note that TolX will apply to the transformed variables. All other
%  fminsearch parameters are unaffected.
%
% Notes:
%
%  Variables which are constrained by both a lower and an upper
%  bound will use a sin transformation. Those constrained by
%  only a lower or an upper bound will use a quadratic
%  transformation, and unconstrained variables will be left alone.
%
%  Variables may be fixed by setting their respective bounds equal.
%  In this case, the problem will be reduced in size for fminsearch.
%
%  The bounds are inclusive inequalities, which admit the
%  boundary values themselves, but will not permit ANY function
%  evaluations outside the bounds.
%
%
% Example usage:
% rosen = @(x) (1-x(1)).^2 + 105*(x(2)-x(1).^2).^2;
%
% fminsearch(rosen,[3 3])     % unconstrained
% ans =
%    1.0000    1.0000
%
% fminsearchbnd(rosen,[3 3],[2 2],[])     % constrained
% ans =
%    2.0000    4.0000

% size checks
xsize = size(x0);
x0 = x0(:);
n=length(x0);

if (nargin<3) || isempty(LB)
  LB = repmat(-inf,n,1);
else
  LB = LB(:);
end
if (nargin<4) || isempty(UB)
  UB = repmat(inf,n,1);
else
  UB = UB(:);
end

if (n~=length(LB)) || (n~=length(UB))
  error 'x0 is incompatible in size with either LB or UB.'
end

% set default options if necessary
if (nargin<5) || isempty(options)
  options = optimset('fminsearch');
end

% stuff into a struct to pass around
params.args = varargin;
params.LB = LB;
params.UB = UB;
params.fun = fun;
params.n = n;

% 0 --> unconstrained variable
% 1 --> lower bound only
% 2 --> upper bound only
% 3 --> dual finite bounds
% 4 --> fixed variable
params.BoundClass = zeros(n,1);
for i=1:n
  k = isfinite(LB(i)) + 2*isfinite(UB(i));
  params.BoundClass(i) = k;
  if (k==3) && (LB(i)==UB(i))
    params.BoundClass(i) = 4;
  end
end

% transform starting values into their unconstrained
% surrogates. Check for infeasible starting guesses.
x0u = x0;
k=1;
for i = 1:n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      if x0(i)<=LB(i)
        % infeasible starting value. Use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(x0(i) - LB(i));
      end
      
      % increment k
      k=k+1;
    case 2
      % upper bound only
      if x0(i)>=UB(i)
        % infeasible starting value. use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(UB(i) - x0(i));
      end
      
      % increment k
      k=k+1;
    case 3
      % lower and upper bounds
      if x0(i)<=LB(i)
        % infeasible starting value
        x0u(k) = -pi/2;
      elseif x0(i)>=UB(i)
        % infeasible starting value
        x0u(k) = pi/2;
      else
        x0u(k) = 2*(x0(i) - LB(i))/(UB(i)-LB(i)) - 1;
        % shift by 2*pi to avoid problems at zero in fminsearch
        % otherwise, the initial simplex is vanishingly small
        x0u(k) = 2*pi+asin(max(-1,min(1,x0u(i))));
      end
      
      % increment k
      k=k+1;
    case 0
      % unconstrained variable. x0u(i) is set.
      x0u(k) = x0(i);
      
      % increment k
      k=k+1;
    case 4
      % fixed variable. drop it before fminsearch sees it.
      % k is not incremented for this variable.
  end
  
end
% if any of the unknowns were fixed, then we need to shorten
% x0u now.
if k<=n
  x0u(k:n) = [];
end

% were all the variables fixed?
if isempty(x0u)
  % All variables were fixed. quit immediately, setting the
  % appropriate parameters, then return.
  
  % undo the variable transformations into the original space
  x = xtransform(x0u,params);
  
  % final reshape
  x = reshape(x,xsize);
  
  % stuff fval with the final value
  fval = feval(params.fun,x,params.args{:});
  
  % fminsearchbnd was not called
  exitflag = 0;
  
  output.iterations = 0;
  output.funcount = 1;
  output.algorithm = 'fminsearch';
  output.message = 'All variables were held fixed by the applied bounds';
  
  % return with no call at all to fminsearch
  return
end

% now we can call fminsearch, but with our own
% intra-objective function.
[xu,fval,exitflag,output] = fminsearch(@intrafun,x0u,options,params);

% undo the variable transformations into the original space
x = xtransform(xu,params);

% final reshape
x = reshape(x,xsize);

% ======================================
% ========= begin subfunctions =========
% ======================================
function fval = intrafun(x,params)
% transform variables, then call original function

% transform
xtrans = xtransform(x,params);

% and call fun
fval = feval(params.fun,xtrans,params.args{:});
end                     % END of INTRAFUN

% ======================================
function xtrans = xtransform(x,params)
% converts unconstrained variables into their original domains

xtrans = zeros(1,params.n);
% k allows soem variables to be fixed, thus dropped from the
% optimization.
k=1;
for i = 1:params.n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      xtrans(i) = params.LB(i) + x(k).^2;
      
      k=k+1;
    case 2
      % upper bound only
      xtrans(i) = params.UB(i) - x(k).^2;
      
      k=k+1;
    case 3
      % lower and upper bounds
      xtrans(i) = (sin(x(k))+1)/2;
      xtrans(i) = xtrans(i)*(params.UB(i) - params.LB(i)) + params.LB(i);
      % just in case of any floating point problems
      xtrans(i) = max(params.LB(i),min(params.UB(i),xtrans(i)));
      
      k=k+1;
    case 4
      % fixed variable, bounds are equal, set it at either bound
      xtrans(i) = params.LB(i);
    case 0
      % unconstrained variable.
      xtrans(i) = x(k);
      
      k=k+1;
  end
end
end                     % END of XTRANSFORM
end                     % END of FMINSEARCHBND




