function drawBoundingBox(VOCopts, I, boundingbox, pyramidIndex)
    for i= 1:size(boundingbox,2),
        currbox = boundingbox(:,i);

        x1 = currbox(1);
        y1 = currbox(2);
        x2 = currbox(3);
        y2 = currbox(4);
        centerx = floor((x2 + x1)/2); 
        centery = floor((y2 + y1)/2);
        width = ((1/VOCopts.pyramidscale) ^ (pyramidIndex-1)) * VOCopts.seconddim * VOCopts.cellsize;
        height = ((1/VOCopts.pyramidscale) ^ (pyramidIndex-1)) * VOCopts.firstdim * VOCopts.cellsize;
        newx1 = max(1,round(centerx - width/2));
        newx2 = min(size(I,2),round(centerx + width/2));
        newy1 = max(1,round(centery - height/2));
        newy2 = min(size(I,1),round(centery + height/2));
        
        I(newy1:newy2,newx1:newx2,2) = 255;
        I(y1:y2,x1:x2,1) = 255;
        image(I);
    end
end