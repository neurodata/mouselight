function [points_transformed] = transform_points(points,path_to_affine,path_to_velocity,velocity_voxel_size,transformation_direction)
    
    curr_path = mfilename('fullpath');
    curr_path = strsplit(curr_path,'/');
    curr_path(end) = [];
    curr_path = strjoin(curr_path, '/');
    addpath([curr_path,'/Functions/avwQuiet/'])
    addpath([curr_path,'/Functions/downsample/'])
    addpath([curr_path,'/Functions/plotting/'])
    addpath([curr_path,'/Functions/textprogressbar/'])

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

    XT = points(:,1);
    YT = points(:,2);
    ZT = points(:,3);


    if strcmpi(transformation_direction,'atlas')
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
        if strcmpi(transformation_direction,'target')
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

    if strcmpi(transformation_direction,'target')

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

    points_transformed = [Atransx Atransy Atransz];

    % deform source to target
    % first thing is to upsample Aphi to Aphi10
    % due to memory issues, I'm going to have to loop through slices,
    % so this will be a bit slow
    %Idef = zeros(destination_shape);
%    F = griddedInterpolant({yIp,xIp,zIp},Ip,interpolation_method,'nearest');
%    Idef = F(Atransy,Atransx,Atransz);
    disp('done applying deformation to points')


end

