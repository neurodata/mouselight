function threeView(x,y,z,I,p) 


if nargin == 1
    I = x;    
    x = 0:size(I,2)-1;
    y = 0:size(I,1)-1;
    z = 0:size(I,3)-1;
    p = [0.5,0.5,0.5];
end
if nargin == 2
    I = x;    
    p = y;
    x = 0:size(I,2)-1;
    y = 0:size(I,1)-1;
    z = 0:size(I,3)-1;
end
if nargin == 3
error('Does not accept 3 arguments')
end
if nargin == 4
p = [0.5,0.5,0.5];
end


clim = double([min(I(:)),max(I(:))]);
if clim(1) == clim(2)
    clim = clim(1) + [-0.5,0.5];
end

if length(size(I))==3
colormap gray
subplot(2,2,1)
imagesc(x,y,I(:,:,round(size(I,3)*p(3))));
axis image
xlabel x
ylabel y
set(gca,'clim',clim)


subplot(2,2,2)
imagesc(z,y,squeeze(I(:,round(size(I,2)*p(2)),:)));
axis image
xlabel z
ylabel y
set(gca,'clim',clim)

subplot(2,2,3)
% imagesc(z,x,squeeze(I(round(size(I,1)/2),:,:)));
% axis image
% xlabel z
% ylabel x


imagesc(x,z,squeeze(I(round(size(I,1)*p(1)),:,:))');
axis image
xlabel x
ylabel z
set(gca,'clim',clim)


colorbar('position',[0.725 0.1 0.05 0.4])
else % color image
    for i = 1 : size(I,4)
        tmp = I(:,:,:,i);
%         I(:,:,:,i) = (I(:,:,:,i) - min(tmp(:)))/(max(tmp(:)) - min(tmp(:)));
        I(:,:,:,i) = (I(:,:,:,i) - mean(tmp(:)))/std(tmp(:))/3 + 0.5;
    end
colormap gray
subplot(2,2,1)
I1 = I(:,:,round(size(I,3)*p(3)),1);
I2 = I(:,:,round(size(I,3)*p(3)),2);
if size(I,4)>2
    I3 = I(:,:,round(size(I,3)*p(3)),3);
else
    I3 = zeros(size(I1));
end
imagesc(x,y,cat(3,I1,I2,I3));
axis image
xlabel x
ylabel y
set(gca,'clim',clim)


subplot(2,2,2)
I1 = squeeze(I(:,round(size(I,2)*p(2)),:,1));
I2 = squeeze(I(:,round(size(I,2)*p(2)),:,2));
if size(I,4)>2
    I3 = squeeze(I(:,round(size(I,2)*p(2)),:,3));
else
    I3 = zeros(size(I1));
end
imagesc(z,y,cat(3,I1,I2,I3));
axis image
xlabel z
ylabel y
set(gca,'clim',clim)

subplot(2,2,3)
% imagesc(z,x,squeeze(I(round(size(I,1)/2),:,:)));
% axis image
% xlabel z
% ylabel x

I1 = squeeze(I(round(size(I,1)*p(1)),:,:,1))';
I2 = squeeze(I(round(size(I,1)*p(1)),:,:,2))';
if size(I,4)>2
    I3 = squeeze(I(round(size(I,1)*p(1)),:,:,3))';
else
    I3 = zeros(size(I1));
end
imagesc(x,z,cat(3,I1,I2,I3));
axis image
xlabel x
ylabel z
set(gca,'clim',clim)

end