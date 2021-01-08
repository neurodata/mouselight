% we want to estimate a polynomial that matches I to J
% we want it to be smooth but spatially varying
% we enforce smoothness with a differential operator
% we use a strategy of block elimination to do so
% really I don't want to do polynomial, I want to do arbitrary basis
%
% I need a function that will update the coeffs one step
% the input needs to be weight, fourier, Basis, and target
%
% I also need to be able to compute the gradient given 
%
clear all;
close all;
fclose all;

addpath /cis/home/dtward/Functions/plotting

sigmaM = 1;
sigmaR = 10000;
a = 15;
% what I want is low frequency to be essentially free
% and high frequency pay a price

a = 100;
sigmaR = a*100;
p = 1; % this looks pretty good

a = 50;
sigmaR = a*100;
p = 2; % this looks pretty good

% 10 is too narrow
a = 10;
sigmaR = a*100;
p = 2; % this looks pretty good

a = 20;
sigmaR = a*100;
p = 2; % this looks pretty good

a = 20;
sigmaR = a*200;
p = 2; % this looks pretty good






% let's load some data in 2D
I = imread('/cis/home/dtward/Documents/exvivohuman_11T/more_blocks/Brain2/histology/down_032/AD_Hip1/Tau/BRC2614 AD Block1 PHF-1 Location 5_corrected.tif');
J = imread('/cis/home/dtward/Documents/exvivohuman_11T/more_blocks/Brain2/histology/down_032/AD_Hip1/LFB/BRC2614 AD Block1 LFB H&E Location 5_corrected.tif');

I = double(I)/255.0;
J = double(J)/255.0;


nI = [size(I,2), size(I,1)];
nJ = [size(J,2), size(J,1)];
dxI = [1,1];
dxJ = [1,1];

xI = (0:nI(1)-1)*dxI(1);
yI = (0:nI(2)-1)*dxI(2);
[XI,YI] = meshgrid(xI,yI);
xJ = (0:nJ(1)-1)*dxJ(1);
yJ = (0:nJ(2)-1)*dxJ(2);
[XJ,YJ] = meshgrid(xJ,yJ);

bias = 1 - 0.5*exp(-((XJ - mean(xJ)).^2 + (YJ - mean(yJ)).^2)/2/100.^2);
J = bsxfun(@times, J, bias);


A = eye(3);
% A(1:2,end) = [-25;5];
B = inv(A);
Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3);
Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3);
for c = 1 : 3;
    F = griddedInterpolant({yI,xI},I(:,:,c),'linear','nearest');
    AI(:,:,c) = F(Ys,Xs);
end






figure;
subplot(2,2,1)
imagesc(AI);
axis image;
title('atlas')
subplot(2,2,2)
imagesc(J)
axis image;
title('target')


%%
% now frequency
fx = (0:nJ(1)-1)/nJ(1)/dxJ(1);
fy = (0:nJ(2)-1)/nJ(2)/dxJ(2);
[FX,FY] = meshgrid(fx,fy);


L = (1.0 - 2.0*a^2*(  (cos(2*pi*FX*dxJ(1)) - 1)/dxJ(1)^2 + (cos(2*pi*FY*dxJ(2)) - 1)/dxJ(2)^2     )).^p;
LL = L.^2;


% figure;
% imagesc(L)

%%
% the basis functions
B = cat(3,ones(size(J,1),size(J,2)),AI);
B = cat(3,B,AI(:,:,1).^2);
B = cat(3,B,AI(:,:,2).^2);
B = cat(3,B,AI(:,:,3).^2);
B = cat(3,B,AI(:,:,1).*AI(:,:,2));
B = cat(3,B,AI(:,:,1).*AI(:,:,3));
B = cat(3,B,AI(:,:,2).*AI(:,:,3));

% transpose
B = reshape(B,size(B,1),size(B,2),1,size(B,3));


%%
% now the cost
% let the coefficient field by
% C
% we say
% min_C (BC - J)^TWW(BC - J)/2/sigmaM^2 + (LC)^T(LC)/2/sigmaR^2
% the gradient gives
% (BC - J)^T W^T W B/sigmaM^2 + C^T L^T L/sigmaR^2 = 0
% or
% C^T B^T W^T W B/sigmaM^2 + C^T L^T L / sigmaR^2 = J^T W^T W B / sigmaM^2
% transpose it
%(B^TW^TWB/sigmaM^2 + L^T L/sigmaR^2 )C = B^T W^T W J/sigmaM^2
%
% so I've basically got two operators
% the fft one and the spatial one
%(A^TA + B^TB )C = R
% R for right hand side
% we approach this using block elimination, exploiting the fact that B is
% easily invertible
% let BC = D
% A^TA C + B^T D = R
%    B C -     D = 0
% now we have two equations
% we can write
% B^{-T}A^TA C + D = B^{-T}R
% = (B^{-T}A^TA B^{-1}  + I) D = B^{-T}R
% now we solve this for D
% then calculate C by inverting B
% if B maps to something without a null space, it can be one dimensoin less
% then B^T B doesn't have an inverse
% but B B^T does, I would need to figure out how to exploit that



% the weights W
% initial guess of coefficients
C = zeros(nJ(2),nJ(1),size(J,3),size(B,4));
% C = ones(size(C))*0.25;
W = ones(size(J,1),size(J,2));
W = 1 + XJ/max(abs(XJ(:)))*0.2;
W(XJ>300 & YJ>200) = 0;


% get the right side
R = bsxfun(@times, bsxfun(@times, B, W.^2), J)/sigmaM^2;
%
applyA = @(C) bsxfun(@times, sum(bsxfun(@times, B, C),4), W)/sigmaM;
AC = applyA(C);
applyAT = @(AC) bsxfun(@times, bsxfun(@times, W,AC), B)/sigmaM;
AAC = applyAT(AC);

applyB = @(C) ifft(ifft(bsxfun(@times, fft(fft(C,[],1),[],2), L ),[],1),[],2,'symmetric')/sigmaR;
BC = applyB(C);
applyBT = @(BC) ifft(ifft(bsxfun(@times, fft(fft(BC,[],1),[],2), L ),[],1),[],2,'symmetric')/sigmaR;
BBC = applyBT(BC);

applyBi = @(C) ifft(ifft(bsxfun(@rdivide, fft(fft(C,[],1),[],2), L ),[],1),[],2,'symmetric')*sigmaR;
applyBTi = @(BC) ifft(ifft(bsxfun(@rdivide, fft(fft(BC,[],1),[],2), L ),[],1),[],2,'symmetric')*sigmaR;

% my op
applyOp = @(D) applyBTi(applyAT(applyA(applyBi(D)))) + D;
D = applyB(C);
BiR = applyBi(R);


% now I should show the constant
coeffs = squeeze(sum(sum(bsxfun(@times, B, J),1),2))/squeeze(sum(sum(bsxfun(@times,B, permute(B,[1,2,4,3])),2),1));
Jhat2 = sum(bsxfun(@times, B, reshape(coeffs,1,1,size(J,3),size(B,4))),4);
% need to put weights in here

for i = 1 : size(coeffs,1)
    for j = 1 : size(coeffs,2)
        C(:,:,i,j) = coeffs(i,j);
    end
end


subplot(2,2,4)
imagesc(Jhat2)
axis image
EM2 = sum(sum(sum((Jhat2 - J).^2,3).*W.^2))/2/sigmaM^2*prod(dxJ);
title(['Constant coeffs, SSE ' num2str(EM2)]);



niter = 10;
C = C*0;
for it = 1 : niter
    
    Jhat = squeeze(sum(bsxfun(@times, C,B),4));
    EM = sum(sum(sum((Jhat - J).^2,3).*W.^2))/2/sigmaM^2*prod(dxJ);
    
    subplot(2,2,3);
    imagesc(Jhat)
    axis image
    title(['Varying coeffs, SSE ' num2str(EM)])
    
%     subplot(2,2,4)
%     imagesc((Jhat - J)/2*3 + 0.5)
%     axis image
%     title('error')
    
    
    % now fft
    Chat = fft(fft(C,[],1),[],2);
    ER = sum(sum(sum(sum(abs(Chat).^2,3),4).*LL))/2/sigmaR^2*prod(dxJ)/(size(LL,1)*size(LL,2));
    E = EM + ER;
    
    disp(['Energy is ' num2str(E)])
    
    C = estimate_coeffs_2d(B,J,W/sigmaM,L/sigmaR,C,10);
    
    drawnow
    
    
    
end

subplot(2,2,4)
imagesc(Jhat2)
axis image
title('Constant coeffs')
EM = sum(sum(sum((Jhat - J).^2,3).*W.^2))/2/sigmaM^2*prod(dxJ);



%%
% once I've got the coeffs, I should be able to estimate the gradient with
% respect to the image
% this will be two steps
% the output with respect to the input basis
% then the input basis with respect to the image