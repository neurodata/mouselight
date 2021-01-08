function frame2Gif(frame,filename,flag,delaytime)
% if flag is 1, then go forwards and backwards
if nargin == 2
    flag = 0;
end
if flag
    frame = [frame(1:end),frame(end-1:-1:2)];
end
if nargin < 4
    delaytime = 0;
end


nFrames = length(frame);
[nRow,nCol,nSlice] = size(frame(1).cdata);
I = uint8(zeros(nRow,nCol*nFrames,nSlice));
for j = 1 : length(frame)
    I(:,(1:nCol) + nCol*(j-1),:) = frame(j).cdata;
end
[Iind,map] = rgb2ind(I,256);
imwrite(reshape(Iind,nRow,nCol,1,nFrames),map,filename,'loopcount',Inf,'delaytime',delaytime);