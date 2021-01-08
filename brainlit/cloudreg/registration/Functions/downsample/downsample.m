% TO DO fix downsample because the version below only works in powers of 2
function [x,y,z,I] = downsample(x,y,z,I,down)


% downsample

if nargin == 2 % just I and down
    I = x;
    down = y;
    x = 1 : size(I,2);
    y = 1 : size(I,1);
    z = 1 : size(I,3);
    
end


if nargin == 4
    down = 2;
end

if all(down==1)
    return;
end



dx = [x(2)-x(1),y(2)-y(1),z(2)-z(1)];
nx = [size(I,2),size(I,1),size(I,3)];




if length(down)==1 && log(down)/log(2) == round(log(down)/log(2))
    % below is only for powers of 2
    for d = 1 : log(down)/log(2)
        % downsample
        if mod(size(I,1),2)
            I = cat(1,I,I(end,:,:));
        end
        if mod(size(I,2),2)
            I = cat(2,I,I(:,end,:));
        end
        if mod(size(I,3),2)
            I = cat(3,I,I(:,:,end));
        end
        I = 0.125*(I(1:2:end,1:2:end,1:2:end) ...
            + I(1:2:end,1:2:end,2:2:end) ...
            + I(1:2:end,2:2:end,1:2:end) ...
            + I(1:2:end,2:2:end,2:2:end) ...
            + I(2:2:end,1:2:end,1:2:end) ...
            + I(2:2:end,1:2:end,2:2:end) ...
            + I(2:2:end,2:2:end,1:2:end) ...
            + I(2:2:end,2:2:end,2:2:end) ) ;
        
        dx = dx*2;
        nx = [size(I,2),size(I,1),size(I,3)];
        
        %     x = x(1:2:end); % is this the right size? start with x(1) and add dx*nx...
        %     y = y(1:2:end);
        %     z = z(1:2:end);
        x = (0 : nx(1)-1)*dx(1) + x(1);
        y = (0 : nx(2)-1)*dx(2) + y(1);
        z = (0 : nx(3)-1)*dx(3) + z(1);
    end
    
else % general downsampling case
    
    
    nxd = floor(nx./down);
    dxd = dx.*down;
    xd = x(1:down(1):nxd(1)*down);
    yd = y(1:down(2):nxd(2)*down);
    zd = z(1:down(3):nxd(3)*down);
    
    Id = zeros(nxd(2),nxd(1),nxd(3));
    for r = 1 : down(2)
        for c = 1 : down(1)
            for s = 1 : down(3)
                Id = Id + I(r:down(2):nxd(2)*down(2), ...
                    c:down(1):nxd(1)*down(1), ...
                    s:down(3):nxd(3)*down(3));
                
            end
        end
    end
    Id = Id/prod(down);
    
    
    I = Id;
    x = xd;
    y = yd;
    z = zd;
    
end