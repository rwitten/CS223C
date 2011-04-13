function [newexamples] = findNewPositives(VOCopts, cls, newgt, newexamples, newimagenumbers,newdetector)

% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

for i=1:length(newgt),
    image = ids{newimagenumbers(i)};
    

    fd = getFeatures(VOCopts,image);
    if newgt(i)>0
        rec=PASreadrecord(sprintf(VOCopts.annopath,image));
        clsinds=strmatch(cls,{rec.objects(:).class},'exact');
        diff=[rec.objects(clsinds).difficult];
        bbox=cat(1,rec.objects(clsinds(~diff)).bbox)';
        curNewExample=findBestFeature(VOCopts, fd, newdetector, ...
            newexamples(i,:),bbox(:,newgt(i)));%, imread(sprintf(VOCopts.imgpath,image)));
    else
        I = imread(sprintf(VOCopts.imgpath,image));
        bbox = [1; 1; size(I,2); size(I,1)];
        curNewExample=findBestNegativeExample(VOCopts, fd, newdetector, ...
            newexamples(i,:),bbox );
    end
        
    if (size(curNewExample,1) > 0) newexamples(i,:) = curNewExample;
    end
end

%function [example] = findBest(VOCopts, fd,detector,)

