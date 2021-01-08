function h = plotbootstrap(x,y,f,p,nbootstrap)
% plot points x and y
% plot fits using function f of the form
% f = @(x,p)
% for parameters p with initial guess p

if nargin == 0
%     close all;
    x = 1:10;
    y = 0.5*x + 1 + randn(size(x)) + randn*20;
%     f = @(x,p) p(1) + p(2)*x;
%     p = [0,1];
%     f = @(x,p) p(1) + p(2)*x + p(3)*x.^2;
%     p = [0,1,0];
    f = @(x,p) p(1) + p(2)*x + p(3)*x.^2 + p(4)*x.^3;
    p = [0,1,0,0];
    % how about this kernel
    n = 5;
    t0 = linspace(min(x),max(x),n);
    f = @(t,p) sum(bsxfun(@times, p(:),  exp(-(bsxfun(@minus,t, t0(:)).^2)/2/2^2)),1);
%     f = @(t,p) sum(bsxfun(@times, p(:),  exp(-(abs(bsxfun(@minus,t, t0(:))))/2/5^2)),1);
    p = zeros(1,n);
end
if nargin < 5
    nbootstrap = 1000;
end

alpha = 0.05; % 90 percent confidence interval
% alpha = 0.1;
% alpha = 0.25;

OPT.MaxFunEvals = 10000;

nextplot = get(gca,'NextPlot');
% first plot the data
h0 = plot(x,y,'o');
hold on;
color = get(h0,'color');
set(h0,'markerfacecolor',color,'markeredgecolor','none');
% keyboard
% now fit a curve with least squares
p0 = fminsearch(@(p)sum((f(x,p)-y).^2),p,OPT);
% plot the fit
nt = 50;
t = linspace(min(x),max(x),nt);
h1 = plot(t,f(t,p0));
set(h1,'color',color);

% now we do our bootstrap resampling
YFit = zeros(nbootstrap,length(t));
for it = 1 : nbootstrap
    % sample
    sample = randi(length(x),size(x));
    X = x(sample);
    Y = y(sample);
    pi = fminsearch(@(p)sum((f(X,p)-Y).^2),p,OPT);
    YFit(it,:) = f(t,pi);  
%     q = quantile(YFit(1:it,:),[alpha 1-alpha],1); % this does linear interpolation
    tmp = sort(YFit(1:it,:));
    ind0 = floor(alpha*(it-1))+1;
    ind1 = ceil((1-alpha)*(it-1))+1;
    q = [tmp(ind0,:);tmp(ind1,:)];
    % plot(t,q)
    v = [[t;q(1,:)],fliplr([t;q(2,:)])]';
    f_ = [1:length(v)];
    try 
        delete(h2);
%         delete(h3);
    end
    h2 = patch('vertices',v,'faces',f_,'facecolor',color,'edgecolor','none','facealpha',0.5);
%     h3 = plot(t,YFit');
    if ~mod(it-1,10)
%         for i = 1 : length(v)
%             text(v(i,1),v(i,2),num2str(i))
%         end
        
%         keyboard
    drawnow;
    end

% if it == 10
%     keyboard
% end
end



% set hold back to original state
set(gca,'NextPlot',nextplot)

% output
h = [h0,h1,h2];

% make things nice
box off
axis square
set(gca,'linewidth',2)
set(gca,'tickdir','out') % like seaborn
set(h,'linewidth',2)
set(gca,'fontsize',12)