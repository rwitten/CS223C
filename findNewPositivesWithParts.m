function [newexamples outgt outimagenumbers newpbboxes] = findNewPositivesWithParts(VOCopts, cls, newgt, newimagenumbers,  detector, pbboxes)
partfirstdim = VOCopts.partfirstdim;
partseconddim = VOCopts.partseconddim;
numparts = VOCopts.numparts;
rootfirstdim = VOCopts.firstdim;
rootseconddim = VOCopts.seconddim;
% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

%Get root filter
curStart = 1;
curEnd = curStart + rootfirstdim*rootseconddim*VOCopts.blocksize^2*VOCopts.numgradientdirections - 1;
rootFilter = detector.multiplier*detector.w(curStart:curEnd);
rootFilter = reshape(rootFilter, [rootfirstdim rootseconddim VOCopts.blocksize^2*VOCopts.numgradientdirections]);

%Get part filter
partFilters = {};
for i=1:numparts
    curStart = curEnd + 1;
    curEnd = curStart + partfirstdim*partseconddim*VOCopts.blocksize^2*VOCopts.numgradientdirections - 1;
    curFilter = detector.multiplier*detector.w(curStart:curEnd);
    partFilters{i} = reshape(curFilter, [round(pbboxes(4,i)-pbboxes(2,i)+1) round(pbboxes(3,i)-pbboxes(1,i)+1) ...
        VOCopts.blocksize^2*VOCopts.numgradientdirections]);
end

%Get deform scores
deformScores = zeros(4,numparts);
for i=1:numparts
    curStart = curEnd + 1;
    curEnd = curStart + 3;
    deformScores(:,i) = detector.multiplier*detector.w(curStart:curEnd)';
end

%Sort imagenumbers
[~,sortIndices] = sort(abs(newimagenumbers));
outgt = [];
newgt = newgt(sortIndices);
outimagenumbers = [];
newimagenumbers = newimagenumbers(sortIndices);
outNum = 1;
curpbboxes = zeros(size(pbboxes));
pbboxesArray = [];
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
        [curNewExample flipExample, curpbboxes]=findBestFeatureWithParts(VOCopts, fd, rootFilter, partFilters,...
            deformScores, pbboxes, bbox(:,newgt(i)));
    else
        [~,curNewExample, flipExample, curpbboxes]=findBestNegativeExampleWithParts(VOCopts, fd,rootFilter, partFilters,...
            deformScores, pbboxes);
    end
    disp 'Image evaluation time'
    toc
    outimagenumbers([outNum outNum+1]) = newimagenumbers(i) * [1; -1];
    outgt([outNum outNum+1]) = [newgt(i); newgt(i)];
    newexamples([outNum outNum+1],:) = [curNewExample; flipExample];
    outNum = outNum+2;
    pbboxesArray(end+1,:,:) = curpbboxes;
end

newpbboxes = squeeze(mean(pbboxesArray,1));
outgt = squeeze(outgt);
outimagenumbers = squeeze(outimagenumbers);

%function [example] = findBest(VOCopts, fd,detector,)

