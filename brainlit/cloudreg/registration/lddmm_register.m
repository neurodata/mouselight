function [] = lddmm_register(in_args)

    arg_list = ["target_name", "registration_prefix", "atlas_prefix", "dxJ0", "grid_correction", "missing_data_correction", "bias_correction", "fixed_scale", "initial_affine", "sigmaR", "num_iterations"];
    defaults = {'~/autofluorescence_data.tif', '~/registration', '~/CloudReg/registration/atlases/', [9.36 9.36  5], 1, 1, 1, 1.0, eye(4), 1e4, 5000};

    if ~exist('in_args','var') || isempty('in_args')
        in_args = defaults;
    end
    
    for i = 1:length(arg_list)
        try
            args.(arg_list(i)) = in_args.(arg_list(i));
        catch
            args.(arg_list(i)) = defaults{i};
        end
    end

    % Determine where your m-file's folder is.
    folder = fileparts(which(mfilename)); 
    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder));

    A = args.initial_affine;
    nT = 10; % number of timesteps over which to integrate flow
    sigmaC = 5.0; % spatial penalty of contrast mapping
    CA = 1; % estimate artficat intensity
    CB = -1; % estimate background intensity

    % order for contrast mapping
    % order 4 means cubic polynomial + constant
    order = 4;
    nM = 1; % do expectation maximization
    nMaffine = 1; % number of m steps per e step durring affine only

    % number of affine only iterations
    naffine = 0;

    % smoothness length scale for velocity field
    % larger means smoother velocity field
    a = 500;

    % gauss newton affine step size
    eA = 0.2;

    % velocity field update step size
    eV = 1e6;

    % prior on brain, artifact, background likelihood (in that order)
    prior = [0.79, 0.2, 0.01];

    % % total number of iterations
    % niter = 5000;

    do_GN = 1; % do gauss newton
    uniform_scale_only = 1; % for uniform scaling
    rigid_only  = 0; % constrain affine to be rigid

    %%%% end parameters %%%%


    downloop_start = 1;
    for downloop = downloop_start : 2

        if downloop > 1
            eV = eV/2;
            niter = 500;
        end

        prefix = args.registration_prefix;

        if downloop == 1
            template_name = strcat(args.atlas_prefix,'/average_template_100.nrrd');
            label_name = strcat(args.atlas_prefix, '/annotation_100.nrrd');

        elseif downloop == 2
            template_name = strcat(args.atlas_prefix,'/average_template_50.nrrd');
            label_name = strcat(args.atlas_prefix, '/annotation_50.nrrd');
            
        end
        
        
        % process some input strings for compatibility with downloop
        if downloop == 1
            vname = ''; % input mat file to restore v, empty string if not restoring
            Aname = ''; % input mat file to restore A, empty string if not restoring
            coeffsname = ''; % input mat file to restore A, empty string if not restoring
        else
            [filepath,name,ext] = fileparts(prefix);
            vname = [filepath,filesep, name,['downloop_' num2str(downloop-1) '_'], ext , 'v.mat'];
            Aname = [filepath,filesep, name,['downloop_' num2str(downloop-1) '_'], ext , 'A.mat'];
            coeffsname = [filepath,filesep, name,['downloop_' num2str(downloop-1) '_'], ext , 'coeffs.mat'];

        end

        % add down loop to prefix
        [filepath,name,ext] = fileparts(prefix);
        prefix = [filepath,filesep, name,['downloop_' num2str(downloop) '_'], ext];
        
        %%
        [filepath,~,~] = fileparts(prefix);
        if ~exist(filepath,'dir')
            mkdir(filepath);
        end
        
        
        %%
        % allen atlas
        [I,meta] = nrrdread(template_name);
        I = double(I);
        dxI = diag(sscanf(meta.spacedirections,'(%d,%d,%d) (%d,%d,%d) (%d,%d,%d)',[3,3]))';
        
        
        
        % want padding of 1mm
        npad = round(1000/dxI(1));
        I = padarray(I,[1,1,1]*npad,0,'both');
        
        
        % scale it for numerical stability, since its scale doesn't matter
        I = I - mean(I(:));
        I = I/std(I(:));
        nxI = [size(I,2),size(I,1),size(I,3)];
        xI = (0:nxI(1)-1)*dxI(1);
        yI = (0:nxI(2)-1)*dxI(2);
        zI = (0:nxI(3)-1)*dxI(3);
        xI = xI - mean(xI);
        yI = yI - mean(yI);
        zI = zI - mean(zI);
        danfigure(1);
        sliceView(xI,yI,zI,I);
        saveas(gcf,[prefix 'example_atlas.png'])
        
        [XI,YI,ZI] = meshgrid(xI,yI,zI);
        fxI = (0:nxI(1)-1)/nxI(1)/dxI(1);
        fyI = (0:nxI(2)-1)/nxI(2)/dxI(2);
        fzI = (0:nxI(3)-1)/nxI(3)/dxI(3);
        [FXI,FYI,FZI] = meshgrid(fxI,fyI,fzI);
        
        [L, meta] = nrrdread(label_name);
        dxL = diag(sscanf(meta.spacedirections,'(%d,%d,%d) (%d,%d,%d) (%d,%d,%d)',[3,3]))';
        L = padarray(L,[1,1,1]*npad,0,'both');
        
        
        %%
        info = imfinfo(args.target_name);
        % downsample to about same res as atlas
        down = round(dxI./args.dxJ0);
        textprogressbar('reading target: ');
        num_slices = length(info);
        for f = 1 : num_slices
            textprogressbar((f/num_slices)*100);
            %disp(['File ' num2str(f) ' of ' num2str(length(info))])
            J_ = double(imread(args.target_name,f));
            if f == 1
                nxJ0 = [size(J_,2),size(J_,1),length(info)];
                nxJ = floor(nxJ0./down);
                J = zeros(nxJ(2),nxJ(1),nxJ(3));
                WJ = zeros(nxJ(2),nxJ(1),nxJ(3));
            end
            % downsample J_
            Jd = zeros(nxJ(2),nxJ(1));
            WJd = zeros(size(Jd)); % when there is no data, we have value 0
            for i = 1 : down(1)
                for j = 1 : down(2)
                    Jd = Jd + J_(i:down(2):down(2)*nxJ(2), j:down(1):down(1)*nxJ(1))/down(1)/down(2);
            WJd = WJd + double((J_(i:down(2):down(2)*nxJ(2), j:down(1):down(1)*nxJ(1))/down(1)/down(2)>0));
                end
            end
            
            slice = floor( (f-1)/down(3) ) + 1;
            if slice > nxJ(3)
                break;
            end
            J(:,:,slice) = J(:,:,slice) + Jd/down(3);
            WJ(:,:,slice) = WJ(:,:,slice) + WJd/down(3);
            
            if ~mod(f-1,10)
                danfigure(1234);
                imagesc(J(:,:,slice));
                axis image
                danfigure(1235);
                imagesc(WJ(:,:,slice));
                axis image	    
                drawnow;
            end
        end
        textprogressbar('done reading target.');
        dxJ = args.dxJ0.*down;
        xJ = (0:nxJ(1)-1)*dxJ(1);
        yJ = (0:nxJ(2)-1)*dxJ(2);
        zJ = (0:nxJ(3)-1)*dxJ(3);
        
        xJ = xJ - mean(xJ);
        yJ = yJ - mean(yJ);
        zJ = zJ - mean(zJ);

        nplot = 5;

        J0 = J; % save it
        J0_orig = J0;
        
        danfigure(2);
        sliceView(xJ,yJ,zJ,J0_orig);
        saveas(gcf,[prefix 'example_target.png'])
        
        %%
        % missing data correction
        if args.missing_data_correction
            WJ = WJ/max(WJ(:));
            q = 0.01;
            c = quantile(J(WJ==1),q);
            J_ = J;
            J_ = J_.*(WJ) + c*(1-WJ);
            danfigure(22)
            sliceView(xJ,yJ,zJ,J_,nplot)
            J0_orig = J_;
        else
            WJ = 1;
        end

        %%
        % grid correction
        if args.grid_correction
            args.grid_correction_blur_width = 150;
            J0 = correct_grid(J0_orig, xJ, yJ, 3, args.grid_correction_blur_width);
            danfigure(3);
            sliceView(xJ,yJ,zJ,J0);
            axis image
            saveas(gcf,[prefix 'example_target_grid.png'])
        end
        
        
        %%
        % basic inhomogeneity correction based on histogaam flow
        % first find a low threshold for taking logs
        J = J0;
        
        range = [min(J(:)), max(J(:))];
        range = mean(range) + [-1,1]*diff(range)/2*1.25;
        
        nb = 300; % better
        
        bins = linspace(range(1),range(2),nb);
        db = (bins(2)-bins(1));
        width = db*2;
        hist_ = zeros(1,nb);
        for b = 1 : nb
            hist_(b) = sum(exp(-(J(:) - bins(b)).^2/2/width^2)/sqrt(2*pi*width^2),1);
        end
        figure;
        plot(bins,hist_)
        thresh = bins(find(hist_==max(hist_),1,'first'))*0.5;
        
        
        
        J(J<thresh) = thresh;
        J = log(J);
        Jbar = mean(J(:));
        Jstd = std(J(:));
        J = J - Jbar;
        J = J/Jstd;
        
        % about 1 mm of padding
        padtemp = round(1000/dxI(1));
        
        J = padarray(J,[1,1,1]*padtemp,'symmetric');
        xJp = (0:size(J,2)-1)*dxJ(1);
        yJp = (0:size(J,1)-1)*dxJ(2);
        zJp = (0:size(J,3)-1)*dxJ(3);
        
        xJp = xJp - mean(xJp);
        yJp = yJp - mean(yJp);
        zJp = zJp - mean(zJp);
        
        
        [XJ,YJ,ZJ] = meshgrid(xJp,yJp,zJp);
        width = 1000;
        K = exp(-(XJ.^2 + YJ.^2 + ZJ.^2)/2/(width)^2);
        K = K / sum(K(:));
        Ks = ifftshift(K);
        Kshat = fftn(Ks);
        
        
        
        % %%
        close all;
        danfigure(14);
        sliceView(xJ,yJ,zJ,exp(J))
        
        
        % iterate
        if args.bias_correction
        
            if args.missing_data_correction
                niterhom = 20;
            else
                niterhom = 10;
            end
            textprogressbar('correcting inhomogeneity: ');
            for it = 1 : niterhom
                textprogressbar((it/niterhom)*100);
                range = [min(J(:)), max(J(:))];
                range = mean(range) + [-1,1]*diff(range)/2*1.25;
                
                bins = linspace(range(1),range(2),nb);
                db = (bins(2)-bins(1));
                width = db*1;
                
                
                hist_ = zeros(1,nb);
                for b = 1 : nb
                    hist_(b) = sum(exp(-(J(:) - bins(b)).^2/2/width^2)/sqrt(2*pi*width^2),1);
                end
                danfigure(10);
                plot(bins,hist_)
                dhist = gradient(hist_,db);
                % now interpolate
                F = griddedInterpolant(bins,dhist,'linear','nearest');
                histgrad = reshape(F(J(:)),size(J));
                % I don't really like this although I do like the sign (tiny slope in flat
                % regions)
                histgrad = sign(histgrad);
                
                danfigure(11);
                sliceView(xJ,yJ,zJ,histgrad,5,[-1,1]);
                histgrad = ifftn(fftn(histgrad).*Kshat,'symmetric');
                danfigure(12);
                sliceView(xJ,yJ,zJ,histgrad);
                
                ep = 1e-1;
                
                
                J = J + ep*histgrad;
                
                % standardize
                J = J - mean(J(:));
                J = J / std(J(:));
                
                danfigure(13);
                sliceView(xJ,yJ,zJ,exp(J))
                
                
            % disp(['Finished it ' num2str(it)])
                drawnow
                
                
                
            end
            textprogressbar('done correcting inhomogeneity');
        end	
        J = exp(J(padtemp+1:end-padtemp,padtemp+1:end-padtemp,padtemp+1:end-padtemp));
        J = J - mean(J(:));
        J = J/std(J(:));
        
        if args.bias_correction
            danfigure(3);
            sliceView(xJ,yJ,zJ,J);
            saveas(gcf,[prefix 'example_target_grid_hom.png'])
        end

        %%
        % set up target grid points included padded grid points for better boundary
        % conditions
        [XJ,YJ,ZJ] = meshgrid(xJ,yJ,zJ);
        
        fxJ = (0:nxJ(1)-1)/nxJ(1)/dxJ(1);
        fyJ = (0:nxJ(2)-1)/nxJ(2)/dxJ(2);
        fzJ = (0:nxJ(3)-1)/nxJ(3)/dxJ(3);
        [FXJ,FYJ,FZJ] = meshgrid(fxJ,fyJ,fzJ);
        
        xJp = [xJ(1)-dxJ(1), xJ, xJ(end)+dxJ(1)];
        yJp = [yJ(1)-dxJ(2), yJ, yJ(end)+dxJ(2)];
        zJp = [zJ(1)-dxJ(3), zJ, zJ(end)+dxJ(3)];
        
        %%
        % now we map them!
        %%
        % get clims for atlas and target
        danfigure(1);
        sliceView(xI,yI,zI,I)
        climI = get(gca,'clim');
        danfigure(2);
        sliceView(xJ,yJ,zJ,J)
        climJ = get(gca,'clim');

        % weight of matching in cost function
        sigmaM = std(J(:));
        % background weight
        sigmaB = sigmaM * 2;
        % artifact weight
        sigmaA = sigmaM * 5;
        
        p = 2;
        apre = 1000;
        ppre = 2;
        aC = 750; % try a little smaller
        pC = 2;
        
        
        % for debugging display dxI
        LL = (1 - 2 * a^2 * ( (cos(2*pi*dxI(1)*FXI) - 1)/dxI(1)^2 + (cos(2*pi*dxI(2)*FYI) - 1)/dxI(2)^2 + (cos(2*pi*dxI(3)*FZI) - 1)/dxI(3)^2 )).^(2*p);
        Khat = 1.0./LL;
        
        
        LLpre = (1 - 2 * apre^2 * ( (cos(2*pi*dxI(1)*FXI) - 1)/dxI(1)^2 + (cos(2*pi*dxI(2)*FYI) - 1)/dxI(2)^2 + (cos(2*pi*dxI(3)*FZI) - 1)/dxI(3)^2 )).^(2*ppre);
        Khatpre = 1.0./LLpre;
        
        LC = (1 - 2 * aC^2 * ( (cos(2*pi*dxJ(1)*FXJ) - 1)/dxJ(1)^2 + (cos(2*pi*dxJ(2)*FYJ) - 1)/dxJ(2)^2 + (cos(2*pi*dxJ(3)*FZJ) - 1)/dxJ(3)^2 )).^(pC);
        LLC = LC.^2;
        iLC = 1.0/LC;
        KhatC = 1.0./LLC;
        
        vtx = zeros([size(I),nT]);
        vty = zeros([size(I),nT]);
        vtz = zeros([size(I),nT]);
        
    
        % load data
        if ~isempty(Aname)
            variables = load(Aname);
            A = variables.A;
        end
        if ~isempty(vname)
            variables = load(vname);
            vtx0 = variables.vtx;
            vty0 = variables.vty;
            vtz0 = variables.vtz;
            % if size does not match we will have to resample
            if size(vtx0,4) ~= size(vtx,4)
                error('Restoring v with different number of timesteps is not supported')
            end
            if any(size(vtx0)~=size(vtx))
                for t = 1 : size(vtx0,4)
                    disp(['Upsampling restored velocity field ' num2str(t) ' of ' num2str(size(vtx0,4))])
                    vtx(:,:,:,t) = upsample(vtx0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
                    vty(:,:,:,t) = upsample(vty0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
                    vtz(:,:,:,t) = upsample(vtz0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
                end
            end
            naffine = 0;
            warning('Because you are restoring a velocity field, we are setting number of affine only steps to 0')
        end
        
        
        It = zeros([size(I),nT]);
        It(:,:,:,1) = I;
        
        
        if ~isempty(coeffsname)
            coeffs = load(coeffsname);
            coeffs = coeffs.coeffs;
            coeffs_1 = upsample(coeffs(:,:,:,1),[size(J,1),size(J,2),size(J,3)]);
            coeffs_2 = upsample(coeffs(:,:,:,2),[size(J,1),size(J,2),size(J,3)]);
            coeffs_3 = upsample(coeffs(:,:,:,3),[size(J,1),size(J,2),size(J,3)]);
            coeffs_4 = upsample(coeffs(:,:,:,4),[size(J,1),size(J,2),size(J,3)]);
            coeffs = cat(4,coeffs_1,coeffs_2,coeffs_3,coeffs_4);
        else
            % we need an initial linear transformation to compute our first weight
            Jq = quantile(J(:),[0.1 0.9]);
            Iq = quantile(I(:),[0.1,0.9]);
            coeffs = [mean(Jq)-mean(Iq)*diff(Jq)/diff(Iq); diff(Jq)/diff(Iq)];
            % if I do a higher order transform, I should just set the nonlinear
            % components to 0
            coeffs = [coeffs;zeros(order-2,1)];
            % note that it might be convenient to work with low order at beginning and
            % then increase order
            % make the coeffs a function of space
            coeffs = reshape(coeffs,1,1,1,[]) .*  ones([size(J),order]);
        end
        
        % start
        Esave = [];
        EMsave = [];
        ERsave = [];
        EAsave = [];
        EBsave = [];
        Asave = [];
        frame_errW = [];
        frame_I = [];
        frame_phiI = [];
        frame_errRGB = [];
        frame_W = [];
        frame_curve = [];
        dt = 1/nT;
        tic
        for it = 1 : niter
            
            % deform image
            phiinvx = XI;
            phiinvy = YI;
            phiinvz = ZI;
            for t = 1 : nT * (it > naffine)
                
                
                % sample image
                if t > 1
                    F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
                    It(:,:,:,t) = F(phiinvy,phiinvx,phiinvz);
                end
                % update diffeo, add and subtract identity for better boundary conditions
                Xs = XI - vtx(:,:,:,t)*dt;
                Ys = YI - vty(:,:,:,t)*dt;
                Zs = ZI - vtz(:,:,:,t)*dt;
                F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
                phiinvx = F(Ys,Xs,Zs) + Xs;
                F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
                phiinvy = F(Ys,Xs,Zs) + Ys;
                F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
                phiinvz = F(Ys,Xs,Zs) + Zs;
                
            end
            F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
            phiI = F(phiinvy,phiinvx,phiinvz);
            danfigure(6791);
            sliceView(xI,yI,zI,phiI)
            
            % now apply affine, go to sampling of J
            % ideally I should fix the double interpolation here
            % for now just leave it
            B = inv(A);
            Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
            Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
            Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
            
            % okay if I did this together I would see
            % AphiI = I(phiinv(B x))
            % first sample phiinv at Bx
            % then sample I at phiinv Bx
            F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
            phiinvBx = F(Ys,Xs,Zs) + Xs;
            F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
            phiinvBy = F(Ys,Xs,Zs) + Ys;
            F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
            phiinvBz = F(Ys,Xs,Zs) + Zs;
            F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
            AphiI = F(phiinvBy,phiinvBx,phiinvBz);
            
            
            % now apply the linear intensity transformation
            % order is 1 plus highest power
            fAphiI = zeros(size(J));
            for o = 1 : order
                fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
            end
            
            
            danfigure(3);
            sliceView(xJ,yJ,zJ,fAphiI,nplot,climJ);
            
            err = fAphiI - J;
            
            danfigure(6666);
            sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
            
            % now a weight
            doENumber = nMaffine;
            if it > naffine; doENumber = nM; end
            if ~mod(it-1,doENumber)
                prior = prior / sum(prior);
                WM = 1/sqrt(2*pi*(sigmaM^2)).*exp(-1.0/2.0/sigmaM^2*err.^2) * prior(1);
                WA = 1/sqrt(2*pi*sigmaA^2)*exp(-1.0/2.0/sigmaA^2*(CA - J).^2) * prior(2);
                WB = 1/sqrt(2*pi*sigmaB^2)*exp(-1.0/2.0/sigmaB^2*(CB - J).^2) * prior(3);
                
                Wsum = WM + WA + WB;
                
                % due to numerical error, there are sum places where Wsum may be 0
                wsm = max(Wsum(:));
                wsm_mult = 1e-6;
                Wsum(Wsum<wsm_mult*wsm) = wsm_mult*wsm;
                
                WM = WM./Wsum;
                WA = WA./Wsum;
                WB = WB./Wsum;
                
                % there is probably a numerically better way to do this
                % if I took logs and then subtractedthe max and then took
                % exponentials maybe
                % not worth it, the numerics are okay
                danfigure(45);
                %         sliceView(xJ,yJ,zJ,WM);
                sliceView(xJ,yJ,zJ,cat(4,WM,WA,WB),nplot);
                
            end
            errW = err.*WM.*WJ;
            
            % now we hit the error with Df, the D is with respect to intensity
            % parameters, not space
            % note f is a map from R to R in this case, so Df is just a scalar
            Df = zeros(size(errW));
            for o = 2 : order
                % note I had a mistake here where I did fAphiI! it is now fixed
                Df = Df + (o-1)*AphiI.^(o-2).*coeffs(:,:,:,o);
            end
            errWDf = errW.*Df;
            
            
            danfigure(4);
            sliceView(xJ,yJ,zJ,errW,nplot);
            
            
            % cost
            vtxhat = fft(fft(fft(vtx,[],1),[],2),[],3);
            vtyhat = fft(fft(fft(vty,[],1),[],2),[],3);
            vtzhat = fft(fft(fft(vtz,[],1),[],2),[],3);
            ER = sum(sum(sum(LL.*sum(abs(vtxhat).^2 + abs(vtyhat).^2 + abs(vtzhat).^2,4))))/2/args.sigmaR^2*dt*prod(dxI)/(size(I,1)*size(I,2)*size(I,3));
            EM = sum(sum(sum((fAphiI - J).^2.*WM)))*prod(dxJ)/2/sigmaM^2;
            EA = sum(sum(sum((CA - J).^2.*WA)))*prod(dxJ)/2/sigmaA^2;
            EB = sum(sum(sum((CB - J).^2.*WB)))*prod(dxJ)/2/sigmaB^2;
            % note regarding energy
            % I also need to include the other terms (related to variance only)
            EM = EM + sum(WM(:)).*log(2*pi*sigmaM^2)/2*prod(dxJ);
            EA = EA + sum(WA(:)).*log(2*pi*sigmaA^2)/2*prod(dxJ);
            EB = EB + sum(WB(:)).*log(2*pi*sigmaB^2)/2*prod(dxJ);
            E = ER + EM + EA + EB;
            fprintf(1,'Iteration %d, energy %g, reg %g, match %g, artifact %g, background %g\n',it,E,ER,EM,EA,EB);
            Esave = [Esave,E];
            ERsave = [ERsave,ER];
            EMsave = [EMsave,EM];
            EAsave = [EAsave,EA];
            EBsave = [EBsave,EB];
            
            % first let's do affine
            % at the end, we'll update affine and coeffs
            % gradient
            [AphiI_x,AphiI_y,AphiI_z] = gradient(AphiI,dxJ(1),dxJ(2),dxJ(3));
            grad = zeros(4,4);
            % NOTE
            % without Gauss Newton, the affine transformation will be updated with
            % rigid transforms.  If the initial guess is nonrigid, it wli lremain
            % nonrigid
            % with GN, the affine transformation will be projected onto rigid
            % transforms, you will lose any nonrigid initialization
            if ~do_GN % do gradient descent
                [AphiI_x,AphiI_y,AphiI_z] = gradient(AphiI,dxJ(1),dxJ(2),dxJ(3));
                grad = zeros(4,4);
                for r = 1 : 3
                    for c = 1 : 4
                        dA = (double((1:4)'==r)) * double(((1:4)==c));
                        AdAB = A * dA * B;
                        AdABX = AdAB(1,1)*XJ + AdAB(1,2)*YJ + AdAB(1,3)*ZJ + AdAB(1,4);
                        AdABY = AdAB(2,1)*XJ + AdAB(2,2)*YJ + AdAB(2,3)*ZJ + AdAB(2,4);
                        AdABZ = AdAB(3,1)*XJ + AdAB(3,2)*YJ + AdAB(3,3)*ZJ + AdAB(3,4);
                        grad(r,c) = -sum(sum(sum(errWDf.*(AphiI_x.*AdABX + AphiI_y.*AdABY + AphiI_z.*AdABZ))))*prod(dxJ)/sigmaM^2;
                    end
                end
                if rigid_only
                    grad(1:3,1:3) = grad(1:3,1:3) - grad(1:3,1:3)';
                end
            else % do Gauss Newton optimization
                [fAphiI_x,fAphiI_y,fAphiI_z] = gradient(fAphiI,dxJ(1),dxJ(2),dxJ(3));
                Jerr = zeros(size(J,1),size(J,2),size(J,3),12);
                count = 0;
                for r = 1 : 3
                    for c = 1 : 4
                        dA = double((1:4==r))' * double((1:4==c));
                        AdAAi = A*dA;
                        Xs = AdAAi(1,1)*XJ + AdAAi(1,2)*YJ + AdAAi(1,3)*ZJ + AdAAi(1,4);
                        Ys = AdAAi(2,1)*XJ + AdAAi(2,2)*YJ + AdAAi(2,3)*ZJ + AdAAi(2,4);
                        Zs = AdAAi(3,1)*XJ + AdAAi(3,2)*YJ + AdAAi(3,3)*ZJ + AdAAi(3,4);
                        count = count + 1;
                        Jerr(:,:,:,count) = (bsxfun(@times, fAphiI_x,Xs) + bsxfun(@times, fAphiI_y,Ys) + bsxfun(@times, fAphiI_z,Zs)).*sqrt(WM);
                    end
                end
                Jerr_ = reshape(Jerr,[],count);
                JerrJerr = Jerr_' * Jerr_;
                % step
                step = JerrJerr \ squeeze(sum(sum(sum(bsxfun(@times, Jerr, err.*sqrt(WM)),3),2),1));
                step = reshape(step,4,3)';
            end % end of affine gradient loop
            
            % now pull back the error, pad it so we can easily get 0 boundary
            % conditions
            errWDfp = padarray(errWDf,[1,1,1],0);
            phi1tinvx = XI;
            phi1tinvy = YI;
            phi1tinvz = ZI;
            % define these variables for output even if only doing affine
            Aphi1tinvx = A(1,1)*phi1tinvx + A(1,2)*phi1tinvy + A(1,3)*phi1tinvz + A(1,4);
            Aphi1tinvy = A(2,1)*phi1tinvx + A(2,2)*phi1tinvy + A(2,3)*phi1tinvz + A(2,4);
            Aphi1tinvz = A(3,1)*phi1tinvx + A(3,2)*phi1tinvy + A(3,3)*phi1tinvz + A(3,4);
            for t = nT*(it>naffine) : -1 : 1
                % update diffeo (note plus)
                Xs = XI + vtx(:,:,:,t)*dt;
                Ys = YI + vty(:,:,:,t)*dt;
                Zs = ZI + vtz(:,:,:,t)*dt;
                F = griddedInterpolant({yI,xI,zI},phi1tinvx-XI,'linear','nearest');
                phi1tinvx = F(Ys,Xs,Zs) + Xs;
                F = griddedInterpolant({yI,xI,zI},phi1tinvy-YI,'linear','nearest');
                phi1tinvy = F(Ys,Xs,Zs) + Ys;
                F = griddedInterpolant({yI,xI,zI},phi1tinvz-ZI,'linear','nearest');
                phi1tinvz = F(Ys,Xs,Zs) + Zs;
                % determinant of jacobian
                [phi1tinvx_x,phi1tinvx_y,phi1tinvx_z] = gradient(phi1tinvx,dxI(1),dxI(2),dxI(3));
                [phi1tinvy_x,phi1tinvy_y,phi1tinvy_z] = gradient(phi1tinvy,dxI(1),dxI(2),dxI(3));
                [phi1tinvz_x,phi1tinvz_y,phi1tinvz_z] = gradient(phi1tinvz,dxI(1),dxI(2),dxI(3));
                detjac = phi1tinvx_x.*(phi1tinvy_y.*phi1tinvz_z - phi1tinvy_z.*phi1tinvz_y) ...
                    - phi1tinvx_y.*(phi1tinvy_x.*phi1tinvz_z - phi1tinvy_z.*phi1tinvz_x) ...
                    + phi1tinvx_z.*(phi1tinvy_x.*phi1tinvz_y - phi1tinvy_y.*phi1tinvz_x);
                
                Aphi1tinvx = A(1,1)*phi1tinvx + A(1,2)*phi1tinvy + A(1,3)*phi1tinvz + A(1,4);
                Aphi1tinvy = A(2,1)*phi1tinvx + A(2,2)*phi1tinvy + A(2,3)*phi1tinvz + A(2,4);
                Aphi1tinvz = A(3,1)*phi1tinvx + A(3,2)*phi1tinvy + A(3,3)*phi1tinvz + A(3,4);
                
                % pull back error with 0 padding
                F = griddedInterpolant({yJp,xJp,zJp},(-errWDfp/sigmaM^2),'linear','nearest');
                lambda = F(Aphi1tinvy,Aphi1tinvx,Aphi1tinvz).*detjac.*abs(det(A));
                
                % get the gradient of the image
                [I_x,I_y,I_z] = gradient(It(:,:,:,t),dxI(1),dxI(2),dxI(3));
                
                % set up the gradient
                gradx = I_x.*lambda;
                grady = I_y.*lambda;
                gradz = I_z.*lambda;
                
                % kernel and reg
                % we add extra smothness here as a predconditioner
                gradx = ifftn((fftn(gradx).*Khat + vtxhat(:,:,:,t)/args.sigmaR^2).*Khatpre,'symmetric');
                grady = ifftn((fftn(grady).*Khat + vtyhat(:,:,:,t)/args.sigmaR^2).*Khatpre,'symmetric');
                gradz = ifftn((fftn(gradz).*Khat + vtzhat(:,:,:,t)/args.sigmaR^2).*Khatpre,'symmetric');
                
                
                % now update
                %         vtx(:,:,:,t) = vtx(:,:,:,t) - gradx*eV;
                %         vty(:,:,:,t) = vty(:,:,:,t) - grady*eV;
                %         vtz(:,:,:,t) = vtz(:,:,:,t) - gradz*eV;
                
                
                % a maximum for stability, this is a maximum but is identify for
                % small argument
                gradxeV = gradx*eV;
                gradyeV = grady*eV;
                gradzeV = gradz*eV;
                norm = sqrt(gradxeV.^2 + gradyeV.^2 + gradzeV.^2);
                mymax = 1*dxJ(1); % is this an appropriate maximum?
                % I do not think there should be a dt here
                % for this data I think 1 voxel is probably way too small
                gradxeV = gradxeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
                gradyeV = gradyeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
                gradzeV = gradzeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
                vtx(:,:,:,t) = vtx(:,:,:,t) - gradxeV;
                vty(:,:,:,t) = vty(:,:,:,t) - gradyeV;
                vtz(:,:,:,t) = vtz(:,:,:,t) - gradzeV;
                
                
            end
            
            
            
            basis = zeros(size(J,1),size(J,2),size(J,3),1,order);
            for o = 1 : order
                basis(:,:,:,1,o) = AphiI.^(o-1);
            end
            if it == 1
                if downloop == 1
                    nitercoeffs = 10;
                else
                    nitercoeffs = 30;
                end
                % vikram testing fewer because maybe better to update slower in the beginning
            else
                nitercoeffs = 5;
                % vikram testing fewer because maybe better to update slower in the beginning
            end
            coeffs = squeeze(estimate_coeffs_3d(basis,J,sqrt(WJ.*WM)/sigmaM,LC/sigmaC,coeffs,nitercoeffs));
            danfigure(466446);
            sliceView(xJ,yJ,zJ,coeffs);
            
            
            % I can also update my constants
            CB = sum(WB(:).*J(:).*WJ(:))/sum(WB(:).*WJ(:));
            CA = sum(WA(:).*J(:).*WJ(:))/sum(WA(:).*WJ(:));
            
            % update A
            if ~do_GN % if gradient descent
                eT = 2e-6;
                eL = 1e-13; % okay seems fine
                post_affine_reduce = 0.1;
                e = [ones(3)*eL,ones(3,1)*eT;0,0,0,0];
                if it > naffine
                    % smaller step size now!
                    e = e * post_affine_reduce;
                end
                A = A * expm(-e.*grad);
            else % do gauss newton
                Ai = inv(A);
                Ai(1:3,1:4) = Ai(1:3,1:4) - eA * step;
                A = inv(Ai);
                if rigid_only
                    [U,S,V] = svd(A(1:3,1:3));
                    A(1:3,1:3) = U * V';
                end
                if uniform_scale_only
                    [U,S,V] = svd(A(1:3,1:3));
                    s = diag(S);
                    s = exp(mean(log(s))) * ones(size(s));
                    if args.fixed_scale ~= 0
                        s = [1,1,1]*args.fixed_scale;
                    end
                    A(1:3,1:3) = U * diag(s) *  V';
            
                end
            end
            
            danfigure(8);
            Asave = [Asave,A(:)];
            subplot(1,3,1)
            plot(Asave([1,2,3,5,6,7,9,10,11],:)')
            title('linear part')
            subplot(1,3,2)
            plot(Asave([13,14,15],:)')
            ylabel um
            title('translation part')
            legend('x','y','z','location','best')
            subplot(1,3,3)
            plot([Esave;ERsave;EMsave;EAsave;EBsave]')
            legend('tot','reg','match','artifact','background','location','best')
            title('Energy')
            saveas(8,[prefix 'energy.png'])
            
            % let's also plot the intensity transformation
            coeffs_ = squeeze(mean(mean(mean(coeffs,1),2),3));
            danfigure(78987);
            t = linspace(climI(1),climI(2),1000)';
            out = t * 0;
            for o = 1 : order
                out = out + coeffs_(o)*t.^(o-1);
            end
            plot(t,out,'linewidth',2)
            set(gca,'xlim',[t(1),t(end)])
            set(gca,'ylim',[min(out),max(out)])
            
            set(gca,'linewidth',2)
            axis square
            set(gca,'fontsize',12)
            xlabel 'Atlas Intensity'
            ylabel 'Target Intensity'
            set(gcf,'paperpositionmode','auto')
            
            
            
            drawnow;
            
            if it <= 100 || ~mod(it-1,11)
                frame_I = [frame_I,getframe(3)];
                frame_errRGB = [frame_errRGB,getframe(6666)];
                frame_W = [frame_W,getframe(45)];
                frame_errW = [frame_errW,getframe(4)];
                frame_curve = [frame_curve,getframe(78987)];
                frame_phiI = [frame_phiI,getframe(6791)];
            end
            
            if it <= 50 || ~mod(it-1,11)
                
                frame2Gif(frame_I, [prefix 'fAphiI.gif']);
                frame2Gif(frame_phiI, [prefix 'phiI.gif']);
                frame2Gif(frame_errRGB,[prefix 'errRGB.gif']);
                frame2Gif(frame_errW,[prefix 'errW.gif'])
                frame2Gif(frame_W,[prefix 'W.gif'])
                frame2Gif(frame_curve,[prefix 'curve.gif'])
                frame2Gif(frame_errRGB(1:10:end),[prefix 'errRGBdown.gif']);
                if it == 1
                    % write J once
                    saveas(2,[prefix 'J.png'])
                    % write I once
                    saveas(1,[prefix 'I.png'])
                end
                % write deformed I every time
                saveas(3,[prefix 'fAphiI.png'])
            end
            
            
            save([prefix 'A.mat'],'A')
            if ~mod(it-1,100)
                save([prefix 'v.mat'],'vtx','vty','vtz','-v7.3')
            end
            
            
        end
        save([prefix 'v.mat'],'vtx','vty','vtz','-v7.3')
        save([prefix 'coeffs.mat'], 'coeffs')
        toc
        
        %%
        % pull back the target
        
        F = griddedInterpolant({yJp,xJp,zJp},padarray(J,[1,1,1]),'linear','nearest');
        Ji = F(Aphi1tinvy,Aphi1tinvx,Aphi1tinvz);
        figure;sliceView(Ji)
        
        % overlay the labels
        rng(1);
        colors = randn(256,3);
        colors(1,:) = 0;
        
        Lm = mod(double(L),256)+1;
        LRGB = reshape([colors(Lm(:),1),colors(Lm(:),2),colors(Lm(:),3)],[size(L),3]);
        figure;sliceView(LRGB)
        
        % opacity of labels
        alpha = 0.125;
        % scale Ji
        Js = (Ji- climJ(1))/diff(climJ);
        Js(Js>1) = 1;
        Js(Js<0) = 0;
        RGB = bsxfun(@plus, LRGB*alpha, Js*(1-alpha));
        close all;
        hf = danfigure();
        set(hf,'paperpositionmode','auto')
        nslices = 8;
        sliceView(xI,yI,zI,RGB,nslices)
        pos = get(hf,'position');
        pos(3) = pos(3)*2;
        set(hf,'position',pos)
        saveas(hf,[prefix 'seg_overlay.png'])
        
        sliceView(xI,yI,zI,LRGB,nslices)
        saveas(hf,[prefix 'seg_only.png'])
        sliceView(xI,yI,zI,Js,nslices)
        saveas(hf,[prefix 'image_only.png'])
        
        
        %%
        % last write out Jdef and L as analyze
        % /*Acceptable values for datatype are*/
        % #define DT_NONE             0
        % #define DT_UNKNOWN          0    /*Unknown data type*/
        % #define DT_BINARY           1    /*Binary             ( 1 bit per voxel)*/
        % #define DT_UNSIGNED_CHAR    2    /*Unsigned character ( 8 bits per voxel)*/
        % #define DT_SIGNED_SHORT     4    /*Signed short       (16 bits per voxel)*/
        % #define DT_SIGNED_INT       8    /*Signed integer     (32 bits per voxel)*/
        % #define DT_FLOAT           16    /*Floating point     (32 bits per voxel)*/
        % #define DT_COMPLEX         32    /*Complex (64 bits per voxel; 2 floating point numbers)/*
        % #define DT_DOUBLE          64    /*Double precision   (64 bits per voxel)*/
        % #define DT_RGB            128    /*A Red-Green-Blue datatype*/
        % #define DT_ALL            255    /*Undocumented*/
        
        
        % write out Ji at this res ( in case the hig hres doesn't work)
        avw = avw_hdr_make;
        avw.hdr.dime.datatype = 16; % 16 bits FLOAT
        avw.hdr.dime.bitpix = 16;
        avw.hdr.dime.dim(2:4) = size(Ji);
        avw.hdr.dime.pixdim([3,2,4]) = dxJ;
        avw.img = Ji;
        avw_img_write(avw,[prefix 'target_to_atlas_low_res_pad.img'])
        
        
    end % of downloop


end