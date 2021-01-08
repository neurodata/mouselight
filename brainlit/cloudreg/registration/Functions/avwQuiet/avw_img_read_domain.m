function [avw,nx,dx,x,y,z,X,Y,Z] = avw_img_read_domain(fileprefix,IMGorient,machine)

% default arguments copied from avw_img_read
if ~exist('fileprefix','var'),
  msg = sprintf('...no input fileprefix - see help avw_img_read\n\n');
  error(msg);
end
if ~exist('IMGorient','var'), IMGorient = ''; end
if ~exist('machine','var'), machine = 'ieee-le'; end

% call avw_img_read
avw = avw_img_read(fileprefix,IMGorient,machine);

% set up domain which I ALWAYS do by hand
% avw.nx = double(avw.hdr.dime.dim([3,2,4]));
% avw.dx = double(avw.hdr.dime.pixdim([3,2,4]));
% avw.x = (0:avw.nx(1)-1)*avw.dx(1);
% avw.y = (0:avw.nx(2)-1)*avw.dx(2);
% avw.z = (0:avw.nx(3)-1)*avw.dx(3);
% 
% [avw.X,avw.Y,avw.Z] = meshgrid(avw.x,avw.y,avw.z);

if nargout > 1
    nx = double(avw.hdr.dime.dim([3,2,4]));
    dx = double(avw.hdr.dime.pixdim([3,2,4]));
end
if nargout > 3
    x = (0:nx(1)-1)*dx(1);
    y = (0:nx(2)-1)*dx(2);
    z = (0:nx(3)-1)*dx(3);
end
if nargout > 6
    [X,Y,Z] = meshgrid(x,y,z);
end