function [C] = estimate_coeffs_3d(B,J,W,L,C,n)
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
%
% since I want to write this as Jhat = B C
% we want B to be size 1,NB and C to be size NB,nJ
% the result should be 1,nJ
% 

% for simplicity of matrix multiplication I want B to be size
% by bx bz 1 bc
% keyboard
Bsize = size(B);
B = reshape(B,Bsize(1),Bsize(2),Bsize(3),1,Bsize(end));
% keyboard
% assume images are rows
Jsize = size(J);
if length(Jsize) == 3
    Jsize(4) = 1;
end
J = reshape(J,Jsize(1),Jsize(2),Jsize(3),1,Jsize(end));

% get the right side of my equation
R = bsxfun(@times, bsxfun(@times, permute(B,[1,2,3,5,4]), W.^2), J);

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
D = D - ep*res*0.5;

disp(sum(res(:).^2))
end

C = applyBi(D,L);

% the output is just BC
% so the gradient is just C


% functions to apply each operator
% note that they are all self adjoint so I don't really need transpose

function AC = applyA(C,B,W)
AC = bsxfun(@times, sum(bsxfun(@times, B, permute(C,[1,2,3,5,4])),5), W);
% first one takes an input size of C, gives an output size of J

function ATAC = applyAT(AC,B,W)
ATAC = permute(bsxfun(@times, bsxfun(@times, W,AC), B),[1,2,3,5,4]);
% second one takes input the size of J and output the size of C

function BC = applyB(C,L)
BC = ifft(ifft(ifft(bsxfun(@times, fft(fft(fft(C,[],1),[],2),[],3), L ),[],1),[],2),[],3,'symmetric');

function C = applyBi(BC,L)
C = ifft(ifft(ifft(bsxfun(@rdivide, fft(fft(fft(BC,[],1),[],2),[],3), L ),[],1),[],2),[],3,'symmetric');

function C = applyBTi(BTC,L)
C = applyBi(BTC,L);

function OC = applyOp(C,B,W,L)
OC = applyBTi(applyAT(applyA(applyBi(C,L),B,W),B,W),L) + C;
