%     % local rotation
%     theta = 50;
%     
%     bx = -3000;
%     by = -1000;
%     
% %    cx = -2000;
%     cx = 500;
%     cy = 3500;
%     
%     rotmat = [cos(theta),-sin(theta);
%         sin(theta),cos(theta)];
%     ROTX = cos(theta*pi/180)*(XI-cx) + sin(theta*pi/180)*(YI-cy) - (XI-cx);
%     ROTY = -sin(theta*pi/180)*(XI-cx) + cos(theta*pi/180)*(YI-cy) - (YI-cy);
%     blob_width = 3000;
%     blob = exp(-((XI - bx).^2 + (YI-by).^2 + (ZI.^2))/2/(blob_width)^2);
%     %x_idx = (XI - bx) > 0;
%     % testing uniform rotation
%     %blob = zeros(size(XI));
%     %blob(x_idx) = 0.5;
%     for t = 1 : nT
%         vty(:,:,:,t) = ROTY.*blob;
%         vtx(:,:,:,t) = ROTX.*blob;
%     end

%    % initial local translation
%   blob_width = 3000;
%   blob_displacement = 3000;
%   bx2 = -5000;
%   by2 = 0;
%   initial_y_disp = exp(-((XI - bx2).^2 + (YI - by2).^2 + (ZI).^2)/2/(blob_width)^2) * blob_displacement;
%   for t = 1 : nT
%       vty(:,:,:,t) = vty(:,:,:,t) + initial_y_disp;
%   end

%%%%%%% for generating images at various fixed scales
if downloop == 1 && fixed_scale == 0 && it == 1
    % show 5 scales
    Asave_ = A;
    [U,S,V] = svd(A(1:3,1:3));
    s = diag(S);
    ss = [0.9,1.0,1.1,1.2,1.3,1.4,1.5];
    for ssloop = 1 : length(ss)
        s = ss(ssloop);
        A(1:3,1:3) = U * diag([s,s,s]) * V';
        B = inv(A);
        Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
        Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
        Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
        
        % okay if I did this together I would see
        % AphiI = I(phiinv(B x))
        % first sample phiinv at Bx
        % then sample I at phiinv Bx
        F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
        phiinvBx = F(Ys,Xs,Zs) + Xs;
        F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
        phiinvBy = F(Ys,Xs,Zs) + Ys;
        F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
        phiinvBz = F(Ys,Xs,Zs) + Zs;
        F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
        AphiI = F(phiinvBy,phiinvBx,phiinvBz);
        
        
        % now apply the linear intensity transformation
        % order is 1 plus highest power
        fAphiI = zeros(size(J));
        for o = 1 : order
            fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
        end
        
        
        danfigure(6666);
        sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
        saveas(6666,[prefix 'test_scale_' num2str(s) '.png']);
    end
    A = Asave_;
    B = inv(A);
    Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
    Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
    Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
    F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
    phiinvBx = F(Ys,Xs,Zs) + Xs;
    F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
    phiinvBy = F(Ys,Xs,Zs) + Ys;
    F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
    phiinvBz = F(Ys,Xs,Zs) + Zs;
    F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
    AphiI = F(phiinvBy,phiinvBx,phiinvBz);
    
    
    % now apply the linear intensity transformation
    % order is 1 plus highest power
    fAphiI = zeros(size(J));
    for o = 1 : order
        fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
    end
    danfigure(6666);
    sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
end