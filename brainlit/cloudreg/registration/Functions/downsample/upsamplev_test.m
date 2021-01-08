% function [vxu,vyu,vzu] = upsamplev(vx,vy,vz)
% for multiscale registration, we will need to upsample v appropriately
% I'd like to do so without changing the energy of the flow
% linear interpolation for example adds sharp corners that really increase
% the energy
% the appropriate way to do this is by zero padding in the fourier domain
% there is a lot of book keeping to deal with though

% start with a simple example
% I'll use 2D
% one dimension even
% one dimension odd
clear all;
close all;
fclose all;
I = image;
I = I.CData;
close all;
I = I(:,1:end-1);
n = [size(I,1),size(I,2)];
figure;
imagesc(I)

nup = [128 128];

% now recall
% if there are an even number of samples
% then we have 0 freq
% paired freqs
% and nyquist (unpaired)
% if we have odd number of samples then we just have 0 freq and paired freq
% so odd is probably easier to deal with
% recall
% fftshift([1,2,3]) = [3,1,2] % zero frequency in the middle
% fftshift([1,2,3,4]) = [3,4,1,2] % nyqust first, zero frequency shifted
% right

% first work out what the fftshift will do, so I can shift it there and
% back
% if its even we just shift right by half
% if its odd we shift by less than half
row_even = ~mod(n(1),2);
col_even = ~mod(n(2),2);

if row_even
    row_shift = n(1)/2;
else
    row_shift = (n(1)-1)/2;
end
if col_even
    col_shift = n(2)/2;
else
    col_shift = (n(2)-1)/2;
end




% now we'll deal with rows
if nup(1)>n(1) % if now upsampling rows, just leave it
    Ihat = fft(I,[],1);
    Ihat = circshift(Ihat,[row_shift,0]);
    if row_even
        % if its even, the first thing I want to do is make nyquist paired
        Ihat(1,:) = Ihat(1,:)/2.0;
        Ihat = [Ihat;Ihat(1,:)];
        n(1) = n(1) + 1;
    end
    % now pad
    Ihat = padarray(Ihat,[nup(1)-n(1),0],0,'post');
    % shift it
    Ihat = circshift(Ihat,[-row_shift,0]);
    % inverse
    I = ifft(Ihat,[],1);
    figure;
    imagesc(I)
    drawnow;
end

% now columns
if nup(2)>n(2)
    Ihat = fft(I,[],2);
    Ihat = circshift(Ihat,[0,col_shift]);
    if col_even
        % if its even, the first thing I want to do is make nyquist paired
        Ihat(:,1) = Ihat(:,1)/2.0;
        Ihat = [Ihat,Ihat(:,1)];
        n(2) = n(2) + 1;
    end
    % now pad
    Ihat = padarray(Ihat,[0,nup(2)-n(2)],0,'post');
    % shift it
    Ihat = circshift(Ihat,[0,-col_shift]);
    % inverse
    I = ifft(Ihat,[],2);
    figure;
    imagesc(I)
    drawnow;
end