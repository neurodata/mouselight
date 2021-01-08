 
function vis_image_and_segs(image_name,seg_name,OPT)

% this function will visualize segmentations overlayed on images
% mandatory arguments are:
%     the name of the image (an analyze, .img file)
%     the name of the segmentation file (an analyze, .img file)     
% and a structure of options
% the various options are specified below under default options
% OPT can be defined with "dot syntax" like
% OPT.dpi = 300;
% OPT.alpha = 0.6; 
% etc.
if nargin < 3
    OPT = struct;
end
if nargin == 1
    error('image filename and segmentation filename are required inputs');
end
if nargin == 0
    % run an example
    warning(['No input arguments specified, running an example'])
    image_name = '/cis/home/dtward/Documents/mricloud_atlases/Adult27-55/Adt27-55_01_Adt27-55_01_MNI.img';
    seg_name = '/cis/home/dtward/Documents/mricloud_atlases/Adult27-55/Adt27-55_01_Adt27-55_01_FullLabels.img';
    OPT = struct;
    OPT.ids = [73:84, 171:180]; % subcortical grey matter sturctures and ventricle
    OPT.nslices = 10;
    OPT.res = 900; % lets use a high resolution
end

% default options
alpha = 0.5; % opacity of labels
ids = []; % a list of segmentation ids to visualize, if empty, do all ids
nslices = 10; % number of evenly space slices to display in each of three orthogonal views
res = 600; % resolution to save figure in dpi
% override defaults if options are specified.
if isfield(OPT,'alpha')
    alpha = OPT.alpha;
end
if isfield(OPT,'ids')
    ids = OPT.ids;
end
if isfield(OPT,'nslices')
    nslices = OPT.nslices;
end
if isfield(OPT,'res')
    res = OPT.res;
end


% load data
addpath /cis/home/dtward/Functions/avwQuiet
[avw,nx,dx,x,y,z] = avw_img_read_domain(image_name);
I = avw.img;

avw = avw_img_read(seg_name);
J = avw.img;

if size(I,1) ~= size(J,1) || size(I,2) ~= size(J,2) || size(J,3) ~= size(J,3)
    error(['Image and segmentations must be the same size, but are size ' sprintf('%dx%dx%d',size(I)) ' and ' sprintf('%dx%dx%d',size(J)) ' (row x col x slice)'])
end




% if you speificied a specific set of ids, just use these

if ~isempty(ids)
    disp('setting ids');
    J_ = zeros(size(J));
    for i = 1 : length(ids)
        J_(J==ids(i)) = ids(i);
    end
    J = J_;
end


% make a color image
rng(1);
colors = hsv(max(J(:))+1);
colors = colors(randperm(length(colors)),:);
% find the biggest id and set it to black
ids = unique(J(:));
for i = 1 : length(ids)
    vol(i) = sum(J(:)==ids(i));
end
largest_id = ids(find(vol == max(vol),1,'first'));
mask = double(J~=largest_id);
R = reshape(colors(J+1,1),size(J));
G = reshape(colors(J+1,2),size(J));
B = reshape(colors(J+1,3),size(J));

% we want alpha to be 0 in background, and specified value in foreground
qlim = [0.01,0.99];
clim = quantile(I(:),qlim);
I = I - clim(1);
I = I/diff(clim);
I(I<0) = 0;
I(I>1) = 1;

image = bsxfun(@plus, I.*(1-alpha*mask) , bsxfun(@times, cat(4,R,G,B),(alpha*mask)));


addpath /cis/home/dtward/Functions/plotting
sliceView(x,y,z,image,nslices);



% save the figure
[~,file1,~] = fileparts(image_name);
[~,file2,~] = fileparts(seg_name);
print(gcf,[file1 '_' file2 '_visualization.png'],'-dpng',sprintf('-r%ddpi',res))
