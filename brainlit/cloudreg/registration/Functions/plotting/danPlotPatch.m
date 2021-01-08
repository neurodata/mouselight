%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   make a plot, but as a surface, 
%
%   inputs x,y,options
%   or y,options
%
%   The options are inputs to a patch
%

function h = danPlotPatch(varargin)

% parse the input arguments
% 
nArg = length(varargin);
if nArg == 0
    error('Must have at least one input argument');
end

% the first argument must be data, either x or y
if ~isa(varargin{1},'numeric')
    error('First argument must be numeric')
end
% nPoints = length(varargin{1});
[nrow, ncol] = size(varargin{1});
if nrow == 1
    nPoints = ncol;
elseif ncol == 1
    nPoints = nrow;
else
    nPoints = nrow;
end

% check if the second argument is numeric
if isa(varargin{2},'numeric')
    % first two arguments are x,y
    x = varargin{1};
    y = varargin{2};
    options = varargin(3:end);
else % second argument is not numeric
    x = (1:nPoints)';
    y = varargin{1};
    options = varargin(2:end);
end


% check for how many lines to draw
[nrowy,ncoly] = size(y);
if nrowy == 1 || ncoly == 1
    % do nothing
    nLines = 1;
    y = y(:); % make sure it is a column
    x = x(:);
else% otherwise use colums of y
    nLines = ncoly;
    
    % check if x is a single line
    [nrowx,ncolx] = size(x);
    if nrowx == 1 || ncolx == 1
        x = x(:);
        x = repmat(x,[1 nLines]);
    % otherwise it's size must match y    
    end
    
end
% keyboard

% options
% defaults = {'facecolor','none','facelighting','none','facealpha',0,'edgecolor','b','edgelighting','none','edgealpha',0.75};
defaults = {'facecolor','none','facelighting','none','facealpha',0,'edgelighting','none','edgealpha',0.75};% note default edge color is taken care of
options = [defaults options];
propName = options(1:2:end);
propVal = options(2:2:end);


colors = lines(nLines);
h = zeros(nLines,1);
for i = 1 : nLines
% h = patch('faces',[1:nPoints],'vertices',[x,y]);
% h = patch('faces',[1:nPoints (nPoints-1):-1:2],'vertices',[x,y]);
% h = patch('faces',[1:nPoints-1;2:nPoints]','vertices',[x,y]);
% h = patch('faces',[1:nPoints],'vertices',[x,y;NaN,NaN]);
% h = patch('faces',[1:nPoints],'vertices',[x,y,ones(nPoints,1)*NaN]);
faces = 1:nPoints+1;
vertices = [x(:,i),y(:,i);NaN,NaN];% avoid closing the polygon


h(i) = patch('faces',faces,'vertices',vertices);
% set(h,propName,propVal)
set(h(i),[{'edgecolor'},propName],[{colors(i,:)},propVal])
% note, specifying a linewidth of 3 or more does antialiasing

end
