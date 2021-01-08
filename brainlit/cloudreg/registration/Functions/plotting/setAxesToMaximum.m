function setAxesToMaximum(h)
if nargin == 0
    h = gcf;
end

hax = findobj(h,'type','axes');
n = length(hax);
xlims = zeros(n,2);
ylims = zeros(n,2);
zlims = zeros(n,2);

for i = 1 : n
    xlims(i,:) = get(hax(i),'xlim');
    ylims(i,:) = get(hax(i),'ylim');
    zlims(i,:) = get(hax(i),'zlim');
end

xlim = [min(xlims(:)),max(xlims(:))];
ylim = [min(ylims(:)),max(ylims(:))];
zlim = [min(zlims(:)),max(zlims(:))];
for i = 1 : n
    set(hax(i),'xlim',xlim,'ylim',ylim,'zlim',zlim)
end