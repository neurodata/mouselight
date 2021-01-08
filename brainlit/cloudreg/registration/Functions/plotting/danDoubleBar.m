function danDoubleBar(landmarkdmean,landmarkdstd,xstring,ystring,legendstring)



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
for i = 1 : nCases/2
    % start at bottom left and go counter clockwise
    h1 = patch( i+barwidth/2*[-1 0 0 -1 -1],landmarkdmean(i)*[0 0 1 1 0] ,'r','edgecolor',edgecolor,'linewidth',barLineWidth+1e-6);
    if ~isempty(landmarkdstd)
    line(i-barwidth/4+errorwidth/2*[-1 1],landmarkdmean(i)+landmarkdstd(i)*[-1 -1],'color','k','linewidth',errorLineWidth)
    line(i-barwidth/4+errorwidth/2*[-1 1],landmarkdmean(i)+landmarkdstd(i)*[1 1],'color','k','linewidth',errorLineWidth)
    line(i-barwidth/4+[0 0],landmarkdmean(i)+landmarkdstd(i)*[-1 1],'color','k','linewidth',errorLineWidth)
    end
    
    j = i+nCases/2;
    h2 = patch( i+barwidth/2*[0 1 1 0 0],landmarkdmean(j)*[0 0 1 1 0] ,'b','edgecolor',edgecolor,'linewidth',barLineWidth+1e-6);
    if ~isempty(landmarkdstd)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[-1 -1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+errorwidth/2*[-1 1],landmarkdmean(j)+landmarkdstd(j)*[1 1],'color','k','linewidth',errorLineWidth)
    line(i+barwidth/4+[0 0],landmarkdmean(j)+landmarkdstd(j)*[-1 1],'color','k','linewidth',errorLineWidth)
    end
end
xlabel(xstring)
ylabel(ystring)
set(gca,'xlim',[0 nCases/2+1])
legend([h1 h2],legendstring,'location','best')
box on