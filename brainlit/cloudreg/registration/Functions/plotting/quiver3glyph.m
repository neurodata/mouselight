function h = quiver3glyph(x,y,z_,u,v_,w)

% first we a single glyph, pointing up in z direction, height 1
% cylinder base
ntheta = 50;
nz = 5;
theta = linspace(0,2*pi,ntheta+1);
theta = theta(1:end-1);
z = linspace(0,0.65,nz);
r = 0.05;
v = [];
f = [];
for i = 1 : nz
    for j = 1 : ntheta
        v = [v; r*cos(theta(j)), r*sin(theta(j)), z(i)];
        if i < nz && j <= ntheta-1
            f = [f;
                [1,2,ntheta+1] + ntheta*(i-1) + (j-1);
                [ntheta+1,2,ntheta+2] + ntheta*(i-1) + (j-1)
                ];
        end
    end
    % last one, closing the circle
    if i < nz
        f = [f;
            [ntheta, 1, ntheta+ntheta]  + ntheta*(i-1) ;
            [ntheta+ntheta, 1, ntheta+1]  + ntheta*(i-1);
            ];
    end
    
end
% close all;
% figure;
% patch('faces',f,'vertices',v,'facecolor','c','edgecolor','k','facelighting','gouraud')
% axis image
% view(30,30)
% light
% drawnow;




% then we need a cone for a hat
foff = size(v,1);
z = linspace(0.65,1,nz);
r0 = 0.1;
r1 = 0;
for i = 1 : nz
    for j = 1 : ntheta
        r_ = r0 - (i-1)/(nz-1)*(r0-r1);
        v = [v; r_*cos(theta(j)), r_*sin(theta(j)), z(i)];
        if i < nz && j <= ntheta-1
            f = [f;
                [1,2,ntheta+1] + ntheta*(i-1) + (j-1) + foff;
                [ntheta+1,2,ntheta+2] + ntheta*(i-1) + (j-1) + foff
                ];
        end
    end
    % last one, closing the circle
    if i < nz
        f = [f;
            [ntheta, 1, ntheta+ntheta]  + ntheta*(i-1) + foff;
            [ntheta+ntheta, 1, ntheta+1]  + ntheta*(i-1) + foff;
            ];
    end
    
end
% figure;
% patch('faces',f,'vertices',v,'facecolor','c','edgecolor','k','facelighting','gouraud')
% axis image
% view(30,30)
% light
% drawnow;


% last a disk
for i = 1 : ntheta-1
    f = [f;
        1,i,i+1];
end
for i = 1 : ntheta-1
    f = [f;
        [1,i,i+1]+foff];
end



% figure;
% patch('faces',f,'vertices',v,'facecolor','c','edgecolor','none','facelighting','gouraud','specularstrength',0)
% axis image
% view(30,30)
% light
% drawnow;




% now loop through
for i = 1 : length(x)
%     keyboard
    % rotate
    
    thetaxy = atan2(v_(i),u(i));
    l1 = sqrt(u(i)^2 + v_(i)^2 );
    l = sqrt(u(i)^2 + v_(i)^2 + w(i)^2 );
    thetaz = asin(  l1 / l  );
    % scale
%     v__ = v*0.25;
%     % scale length
%     v__(:,3) = v__(:,3)*l;
    
    % don't like that
    if l < 1
        v__ = v*l;
    else
        v__ = v;
        v__(:,3) = v__(:,3)*l;
    end
    
    % first toward the x axis
    v__ = ([cos(thetaz), 0, sin(thetaz);
        0 1 0;
        -sin(thetaz),0,cos(thetaz)]*v__')'; 
    % around the z axis
    v__ = ([cos(thetaxy),-sin(thetaxy),0;
        sin(thetaxy),cos(thetaxy),0;
        0 0 1]*v__')';
    v__ = bsxfun(@plus,v__,[x(i),y(i),z_(i)]);
    h(i) = patch('faces',f,'vertices',v__,'facecolor','c','edgecolor','none','specularstrength',0);
end

