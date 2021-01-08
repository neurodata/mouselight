function h = subplotdan(nrows,ncols,number,gaplr,gapud)

if nargin == 4
    gapud = gaplr;
end
if nargin == 3
    gaplr = 0;
    gapud = 0;
end

% the whole width: 1 = gaplr*(ncols+1) + ncols*width
width = (1 - gaplr*(ncols+1) ) / ncols;
height = (1 - gapud*(nrows+1) ) / nrows;

% now we find the position of this plot
% row = ceil( (number-1) / (ncols-1) );
row = ceil( (number-1*0) / (ncols-1*0) );

col = mod(number - 1, ncols) + 1;
if number == 1
    row = 1;
    col = 1;
end
if number == nrows*ncols
    row = nrows;
    col = ncols;
end

% now we convert to coordinates
xpos = gaplr*col + width*(col-1);
ypos = gapud*row + height*(row-1);

% and remember that col is measuring from the top, but position is measuring from the bottom 
ypos = 1 - ypos - height;

% create the subplot
h = subplot('position',[xpos ypos width height]);
