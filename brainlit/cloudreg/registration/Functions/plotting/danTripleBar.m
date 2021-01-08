function danTripleBar(landmarkdmean,landmarkdstd,xstring,ystring,legendstring)



nCases = length(landmarkdmean);
barwidth = 0.8;
errorwidth = 0.1;


barLineWidth = 0;
errorLineWidth = 2;

if barLineWidth==0
    edgecolor = 'none';
else
    edgecolor = 'k';
end

hold on;
for i = 1 : nCases/3
    % start at bottom left and go counter clockwise
    h1 = patch( i+barwidth*(1/3*[0 1 1 0 0]-1/3),landmarkdmean(i)*[0 0 1 1 0] ,'r','edgecolor',edgecolor,'linewidth',barLineWidth+1e-6);
    if ~isempty(landmarkdstd)
    line(i-barwidth/4+errorwidth/2*[-1 1],landmarkdmean(i)+landmarkdstd(i)*[-1 -1],'color','k','linewidth',errorLineWidth)
    line(i-barwidth/4+errorwidth/2*[-1 1],landmarkdmean(i)+landmarkdstd(i)*[1 1],'color','k','linewidth',errorLineWidth)
    line(i-barwidth/4+[0 0],landmarkdmean(i)+landmarkdstd(i)*[-1 1],'color','k','linewidth',errorLineWidth)
    end
    
    j = i+nCases/3;
    h2 = patch( i+barwidth*(1/3*[0 1 1 0 0]-0),landmarkdmean(j)*[0 0 1 1 0] ,'b','edgecolor',edgecolor,'linewidth',barLineWidth+1e-6);
    if ~isempty(landmarkdstd)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[-1 -1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[1 1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+[0 0],landmarkdmean(j)+landmarkdstd(j)*[-1 1],'color','k','linewidth',errorLineWidth)
    end

    j = i+2*nCases/3;
    h3 = patch(  i+barwidth*(1/3*[0 1 1 0 0]+1/3),landmarkdmean(j)*[0 0 1 1 0] ,'g','edgecolor',edgecolor,'linewidth',barLineWidth+1e-6);
    if ~isempty(landmarkdstd)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[-1 -1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[1 1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+[0 0],landmarkdmean(j)+landmarkdstd(j)*[-1 1],'color','k','linewidth',errorLineWidth)
    end

    
end
xlabel(xstring)
ylabel(ystring)
set(gca,'xlim',[0 nCases/3+1])
legend([h1 h2 h3],legendstring,'location','best')
box on