 
function num = danfigure(num)

try
set(0,'currentfigure',num);
catch
    try
figure(num);
    catch
        num = figure;
    end
end