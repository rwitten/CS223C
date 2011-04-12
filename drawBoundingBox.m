function drawBoundingBox(I, boundingbox)
    figure();
    for i= 1:size(boundingbox,2),
        currbox = boundingbox(:,i);

        x1 = currbox(1);
        y1 = currbox(2);
        x2 = currbox(3);
        y2 = currbox(4);
        
        I(y1:y2,x1:x2,mod(i,3)) = 255;
        image(I);
    end
end