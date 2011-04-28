function [newexamples newgt newimagenumbers] = findNewPositives(VOCopts, cls, newgt, newexamples, newimagenumbers, newdetector)

% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

rootFilter = reshape(newdetector.w(1:end-1), [VOCopts.firstdim VOCopts.seconddim VOCopts.blocksize^2*VOCopts.numgradientdirections]);
rootFilter = rootFilter.*newdetector.multiplier;


outgt = [];
outimagenumbers = [];
outNum = 1;
for i=1:length(newgt),
    if (newimagenumbers(i) < 0) 
        continue;
    end
    image = ids{newimagenumbers(i)};
    fd = getFeatures(VOCopts,image);
    tic
    if newgt(i) > 0
        rec=PASreadrecord(sprintf(VOCopts.annopath,image));
        clsinds=strmatch(cls,{rec.objects(:).class},'exact');
        diff=[rec.objects(clsinds).difficult];
        bbox=cat(1,rec.objects(clsinds(~diff)).bbox)';
        [curNewExample flipExample]=findBestFeature(VOCopts, fd, rootFilter, bbox(:,newgt(i)));
    else
        [~,curNewExample, flipExample]=findBestNegativeExample(VOCopts, fd,rootFilter);
    end
    disp 'Image evaluation time'
    toc
    outimagenumbers([outNum outNum+1]) = newimagenumbers(i) * [1; -1];
    outgt([outNum outNum+1]) = [newgt(i); newgt(i)];
    newexamples([outNum outNum+1],:) = [curNewExample; flipExample];
    outNum = outNum+2;
end

outgt = squeeze(outgt);
outimagenumbers = squeeze(outimagenumbers);

%function [example] = findBest(VOCopts, fd,detector,)

