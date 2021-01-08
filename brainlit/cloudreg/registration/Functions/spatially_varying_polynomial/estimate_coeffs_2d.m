function [C] = estimate_coeffs_2d(B,J,W,L,C,n)
% we match basis to image J
% I don't know if I should bother wrapping up
% TO DO
% B basis functions (e.g. polynomial)
% J target image
% W spatially diagonal operator
% L Fourier diagonal operator
% C current guess of coefficients
% n update n times
% note W and L should contain and parameters lke sigma and W

% for simplicity of matrix multiplication I want B to be size
% by bx 1 bc
% keyboard

Bsize = size(B);
B = reshape(B,Bsize(1),Bsize(2),1,Bsize(end));


% get the right side of my equation
R = bsxfun(@times, bsxfun(@times, B, W.^2), J);

% reformulate with my block elimination approach
D = applyB(C,L);
BiR = applyBi(R,L);

for i = 1 : n
% find residual
OpD = applyOp(D,B,W,L);
res = OpD - BiR;

% compute the optimal step size
Ores = applyOp(res,B,W,L);
ep = sum(res(:).^2)/sum(Ores(:).*res(:));

% update
D = D - ep*res;
end

C = applyBi(D,L);

% the output is just BC
% so the gradient is just C


% functions to apply each operator
% note that they are all self adjoint so I don't really need transpose

function AC = applyA(C,B,W)
AC = bsxfun(@times, sum(bsxfun(@times, B, C),4), W);
% should be the size of the output image

function ATAC = applyAT(AC,B,W)
ATAC = bsxfun(@times, bsxfun(@times, W,AC), B);
% should be the size of the coeffs

function BC = applyB(C,L)
BC = ifft(ifft(bsxfun(@times, fft(fft(C,[],1),[],2), L ),[],1),[],2,'symmetric');

function C = applyBi(BC,L)
C = ifft(ifft(bsxfun(@rdivide, fft(fft(BC,[],1),[],2), L ),[],1),[],2,'symmetric');

function C = applyBTi(BTC,L)
C = ifft(ifft(bsxfun(@rdivide, fft(fft(BTC,[],1),[],2), L ),[],1),[],2,'symmetric');

function OC = applyOp(C,B,W,L)
OC = applyBTi(applyAT(applyA(applyBi(C,L),B,W),B,W),L) + C;
% first we smooth
% then we predict image (size of target image)
% then we apply A^T, we end up with size of C
% the we smooth again
% then add C again