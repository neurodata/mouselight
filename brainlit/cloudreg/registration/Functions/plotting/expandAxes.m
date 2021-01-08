function [xlim,ylim,zlim] = expandAxes(factor,h)

if nargin == 0
    factor = 1.5;
end
if nargin == 1
    h = gca;
end


xlim = get(h,'xlim');
ylim = get(h,'ylim');
zlim = get(h,'zlim');

xlim = mean(xlim) + [-1 1]/2*diff(xlim)*factor;
ylim = mean(ylim) + [-1 1]/2*diff(ylim)*factor;
zlim = mean(zlim) + [-1 1]/2*diff(zlim)*factor;

set(h,'xlim',xlim,'ylim',ylim,'zlim',zlim);