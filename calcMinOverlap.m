function minOverlap = calcMinOverlap(bbox, curbox)
    bbx1 = bbox(1);
    bby1 = bbox(2);
    bbx2 = bbox(3);
    bby2 = bbox(4);
    
    curx1 = curbox(1);
    cury1 = curbox(2);
    curx2 = curbox(3);
    cury2 = curbox(4);
   
    
    overlapSize = rectint([bbx1 bby1 (bbx2-bbx1) (bby2-bby1)], [curx1 cury1 (curx2-curx1) (cury2-cury1)]);
    curSize = abs((curx2 - curx1)*(cury2 - cury1));
    bbSize = abs((bbx2 - bbx1) * (bby2 - bby1));
    
    minOverlap = overlapSize/max(bbSize, curSize);
end