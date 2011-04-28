function [  ] = drawWithBB( I,BB,name )
%DRAWWITHBB Summary of this function goes here
%   Detailed explanation goes here

for k = 1:size(BB,2)
    pixelBox=BB(:,k);
    for x=1:size(I,1),
        for y=1:size(I,2),
            if x > pixelBox(2) && x < pixelBox(4) && y>pixelBox(1) && y<pixelBox(3)
                I(x,y,1) = 1e4;
            end
        end
    end
end

if size(BB,2)>0,
    imwrite(I, name, 'png');
end

end

