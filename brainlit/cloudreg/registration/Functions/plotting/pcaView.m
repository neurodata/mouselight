function [az, el, n] = pcaView(X,display);
% output azimuth, elevation, and normal vector

if nargin == 0
    error('Must have at least one input argument');
end
if nargin == 1
    display = 0;
end

% load point data
if ndims(X) ~= 2
    error('Input data must be a 2D array')
end

% check size
[nRow,nCol] = size(X);
if nRow ~= 3
    X = X';
    [nRow,nCol] = size(X);
end
if nRow ~= 3
    error('Input data should represent 3D points')
end

% number of points
n = nCol;

% keyboard
% find the covariance
Sigma = 1/n * (X * X');

% find the eigenvectors
[V,D] = eig(Sigma);
d = diag(D);
[d,permutation] = sort(d,'descend');
V = V(:,permutation);

% we want to set the camera to the direction of the smallest eigenvector
n = V(:,3);


% convert to spherical coordinates
rxy = norm(n(1:2));
%theta = acos(n(3)/rxy);% angle from z, this is wrong
theta = atan2(rxy,n(3));% angle from z
phi = atan2(n(2),n(1));% angle from x

% we set the view in terms of azimuth and elevation
% note azimuth is zero when n is in the -y direction
el = (pi/2-theta)*180/pi;
az = (phi+pi/2)*180/pi;



if display
    subplot(2,2,1)
    scatter3(X(1,:),X(2,:),X(3,:))
    axis image;
    xlabel x
    ylabel y
    zlabel z
    view(0,0);
    
    subplot(2,2,2)
    scatter3(X(1,:),X(2,:),X(3,:))
    axis image;
    xlabel x
    ylabel y
    zlabel z
    view(0,90);
    
    subplot(2,2,3)
    scatter3(X(1,:),X(2,:),X(3,:))
    axis image;
    xlabel x
    ylabel y
    zlabel z
    view(90,0);
    
    subplot(2,2,4)
    scatter3(X(1,:),X(2,:),X(3,:))
    axis image;
    xlabel x
    ylabel y
    zlabel z
    view(az,el)
end
%
