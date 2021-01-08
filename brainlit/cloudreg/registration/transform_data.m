function [] = transform_data(path_to_source,source_voxel_size,path_to_affine,path_to_velocity,velocity_voxel_size,destination_voxel_size,destination_shape,transformation_direction,path_to_output,interpolation_method)
    
    addpath ./Functions/avwQuiet/
    addpath ./Functions/downsample/
    addpath ./Functions/plotting/
    addpath ./Functions/textprogressbar/

    % typical use case 1
    % eg.  transform the allen atlas to the high resolution CLARITY space

    % load the transformations
    A = load(path_to_affine);
    v = load(path_to_velocity);

    % extract transformations from variables
    A = A.A;
    vtx = v.vtx;
    vty = v.vty;
    vtz = v.vtz;

    % predefined params
    % number of timesteps
    nT = size(vtx,4);
    dt = 1/nT;

    % original voxel spacing of image
    dxI0 = source_voxel_size;
    info = imfinfo(path_to_source);

    % load source data and downsample
    % to about same res as target
    down = round(destination_voxel_size./dxI0)
    textprogressbar('reading source: ')
    for f = 1 : length(info)
        %disp(['File ' num2str(f) ' of ' num2str(length(info))])
        textprogressbar((f/length(info))*100)
        I_ = double(imread(path_to_source,f));
        if f == 1
            nxI0 = [size(I_,2),size(I_,1),length(info)];
            nxI = floor(nxI0./down);
            I = zeros(nxI(2),nxI(1),nxI(3));
        end
        % downsample J_
        Id = zeros(nxI(2),nxI(1));
        for i = 1 : down(1)
            for j = 1 : down(2)
                % if im reading labels, don't average here
                Id = Id + I_(i:down(2):down(2)*nxI(2), j:down(1):down(1)*nxI(1))/down(1)/down(2);
            end
        end

        slice = floor( (f-1)/down(3) ) + 1;
        if slice > nxI(3)
            break;
        end
        I(:,:,slice) = I(:,:,slice) + Id/down(3);

    end
    textprogressbar('-- done reading source')

    % update dxI info
    dxI = dxI0.*down;
    xI = (0:nxI(1)-1)*dxI(1);
    yI = (0:nxI(2)-1)*dxI(2);
    zI = (0:nxI(3)-1)*dxI(3);

    xI = xI - mean(xI);
    yI = yI - mean(yI);
    zI = zI - mean(zI);

    xIp = [xI(1)-dxI(1), xI, xI(end)+dxI(1)];
    yIp = [yI(1)-dxI(2), yI, yI(end)+dxI(2)];
    zIp = [zI(1)-dxI(3), zI, zI(end)+dxI(3)];
    Ip = padarray(I,[1,1,1]);

    climI = [min(I(:)),max(I(:))];

    % initialize transformations with voxel starting positions
    % 1D array of voxel locations
    % for each dimension of velocity field
    nxV = [size(vtx,2) size(vtx,1) size(vtx,3)];
    dxV = velocity_voxel_size;
    xV = (0:nxV(1)-1)*dxV(1);
    yV = (0:nxV(2)-1)*dxV(2);
    zV = (0:nxV(3)-1)*dxV(3);
    xV = xV - mean(xV);
    yV = yV - mean(yV);
    zV = zV - mean(zV);
    [XV,YV,ZV] = meshgrid(xV,yV,zV);


    % target voxel spacing
    % and voxel locations
    % for case 1:
    % dxT: voxel size of the clarity image
    % nxT: number of voxels in each dimension
    % dxT,nxT should be in X,Y,Z order not row,col,slice
    % also dxT and nxT should correspond
    dxT = destination_voxel_size
    %  dont switch  here
    nxT = destination_shape
    %nxT = [destination_shape(2) destination_shape(1) destination_shape(3)]

    xT = (0:nxT(1)-1)*dxT(1);
    yT = (0:nxT(2)-1)*dxT(2);
    zT = (0:nxT(3)-1)*dxT(3);

    xT = xT - mean(xT);
    yT = yT - mean(yT);
    zT = zT - mean(zT);

    [XT,YT,ZT] = meshgrid(xT,yT,zT);


    if strcmpi(transformation_direction,'target')
        timesteps = (1: nT);
        indicator = -1;

    else
        timesteps = (nT: -1: 1);
        indicator = 1;
    end

	transx = XV;
	transy = YV;
	transz = ZV;
	
    textprogressbar('integrating velocity field: ')
    for t = timesteps
        if strcmpi(transformation_direction,'atlas')
            textprogressbar(((nT-t+1)/nT)*100)
        else
            textprogressbar((t/nT)*100)
        end

        % update diffeomorphism
        Xs = XV + indicator*vtx(:,:,:,t)*dt;
        Ys = YV + indicator*vty(:,:,:,t)*dt;
        Zs = ZV + indicator*vtz(:,:,:,t)*dt;
        F = griddedInterpolant({yV,xV,zV},transx-XV,'linear','nearest');
        transx = F(Ys,Xs,Zs) + Xs;
        F = griddedInterpolant({yV,xV,zV},transy-YV,'linear','nearest');
        transy = F(Ys,Xs,Zs) + Ys;
        F = griddedInterpolant({yV,xV,zV},transz-ZV,'linear','nearest');
        transz = F(Ys,Xs,Zs) + Zs;
        
    end

    textprogressbar('-- done integrating velocity field')
    clear F

    if strcmpi(transformation_direction,'atlas')

        Atransx = A(1,1)*transx + A(1,2)*transy + A(1,3)*transz + A(1,4);
        Atransy = A(2,1)*transx + A(2,2)*transy + A(2,3)*transz + A(2,4);
        Atransz = A(3,1)*transx + A(3,2)*transy + A(3,3)*transz + A(3,4);

        Fx = griddedInterpolant({yV,xV,zV},Atransx,'linear','nearest');
        Fy = griddedInterpolant({yV,xV,zV},Atransy,'linear','nearest');
        Fz = griddedInterpolant({yV,xV,zV},Atransz,'linear','nearest');

        Atransx = Fx(YT,XT,ZT);
        Atransy = Fy(YT,XT,ZT);
        Atransz = Fz(YT,XT,ZT);


        
    else
        B = inv(A);
        Xs = B(1,1)*XT + B(1,2)*YT + B(1,3)*ZT + B(1,4);
        Ys = B(2,1)*XT + B(2,2)*YT + B(2,3)*ZT + B(2,4);
        Zs = B(3,1)*XT + B(3,2)*YT + B(3,3)*ZT + B(3,4);
        
        % the resulting transformations are  the shape of the target
		F = griddedInterpolant({yV,xV,zV},transx-XV,'linear','nearest');
		Atransx = F(Ys,Xs,Zs) + Xs;
		F = griddedInterpolant({yV,xV,zV},transy-YV,'linear','nearest');
		Atransy = F(Ys,Xs,Zs) + Ys;
		F = griddedInterpolant({yV,xV,zV},transz-ZV,'linear','nearest');
		Atransz = F(Ys,Xs,Zs) + Zs;

    end

    % deform source to target
    % first thing is to upsample Aphi to Aphi10
    % due to memory issues, I'm going to have to loop through slices,
    % so this will be a bit slow
    %Idef = zeros(destination_shape);
    F = griddedInterpolant({yIp,xIp,zIp},Ip,interpolation_method,'nearest');
    Idef = F(Atransy,Atransx,Atransz);
    disp('done applying deformation to source image')

    %textprogressbar('applying deformation to source: ')
    %for i = 1:length(info)
    %    textprogressbar((i/size(Idef,3))*100)
    %    %disp(['Applying deformation slice ' num2str(i) ' of ' num2str(size(Jdef,3))]);
    %	Aphi10x = Fx(YI10_,XI10_,ones(size(XI10_))*zI10(i));
    %	Aphi10y = Fy(YI10_,XI10_,ones(size(XI10_))*zI10(i));
    %	Aphi10z = Fz(YI10_,XI10_,ones(size(XI10_))*zI10(i));
    %
    %	Jdef(:,:,i) = F(Aphi10y,Aphi10x,Aphi10z);
    %end
    %textprogressbar('-- done applying deformation to source')

    % permute dimensions of Idef so it matches
    % C order instead  of F order
    Idef = permute(Idef,[2 1 3]);
    destination_voxel_size = destination_voxel_size([2 1 3])
    avw = avw_hdr_make;
    avw.hdr.dime.datatype = 4; % 16 bits
    avw.hdr.dime.bitpix = 16;
    avw.hdr.dime.dim(2:4) = size(Idef);
%     avw.hdr.dime.pixdim([3,2,4]) = dxJ;
    avw.hdr.dime.pixdim([3,2,4]) = destination_voxel_size;
    avw.img = Idef;
    avw_img_write(avw,path_to_output)

end

