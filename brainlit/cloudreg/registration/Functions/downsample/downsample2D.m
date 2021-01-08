function [x,y,I] = downsample2D(x,y,I,down)


% downsample
if nargin == 3
    down = 2;
end
dx = [x(2)-x(1),y(2)-y(1)];
nx = [size(I,2),size(I,1)];

if length(down)==1 && log(down)/log(2) == round(log(down)/log(2))

for d = 1 : log(down)/log(2)
    % downsample
    if mod(size(I,1),2)
        I = cat(1,I,I(end,:,:));
    end
    if mod(size(I,2),2)
        I = cat(2,I,I(:,end,:));
    end
    I = 0.25*(I(1:2:end,1:2:end) ...
        + I(1:2:end,1:2:end) ...
        + I(1:2:end,2:2:end) ...
        + I(1:2:end,2:2:end) );
    
    dx = dx*2;
    nx = [size(I,2),size(I,1)];
        
    x = (0 : nx(1)-1)*dx(1) + x(1);
    y = (0 : nx(2)-1)*dx(2) + y(1);
end

else


 nxd = floor(nx./down);
    dxd = dx.*down;
    xd = x(1:down(1):nxd(1)*down(1));
    yd = y(1:down(2):nxd(2)*down(2));
    
    Id = zeros(nxd(2),nxd(1));
    for r = 1 : down(2)
        for c = 1 : down(1)
            Id = Id + I(r:down(2):nxd(2)*down(2), ...
                c:down(1):nxd(1)*down(1));
            
        end
    end
    Id = Id/prod(down);
    
    
    I = Id;
    x = xd;
    y = yd;
end