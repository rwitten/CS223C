function example_detector

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

VOCopts.blocksize=2;
VOCopts.cellsize = 8;
VOCopts.numgradientdirections = 9;
VOCopts.firstdim = 10;
VOCopts.seconddim=6;
VOCopts.rootfilterminingiters=5;
VOCopts.seconddim= 6;
VOCopts.rootfilterminingiters=2;
VOCopts.rootfilterupdateiters=0;
VOCopts.NEG_TRAIN_IMAGES=10;
VOCopts.POS_TRAIN_IMAGES=10;
VOCopts.MAX_NEG_EXAMPLES=500;  %this is the size of cache in terms of number of images
VOCopts.MAX_POS_EXAMPLES=500;
VOCopts.HARDNESS_CUTOFF=-0.2; %all neg examples with scores greater than this are kept
VOCopts.pyramidscale = 1/1.1;
VOCopts.hognormclip = 0.6;
VOCopts.rootsamples = 30;
VOCopts.partfirstdim = 6;
VOCopts.partseconddim = 5;
VOCopts.numparts = 6;
VOCopts.TEST_IMAGES = 10;
%VOCopts.firstdim = 32; %empirical average!
%VOCopts.seconddim=22; %empirical average!
% train and test detector for each class
cls='person';

%Load up annotaitons
try
    load('VOCannotations.mat','VOCannotations');
catch
    ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');
    VOCannotations = struct([]);
    for i=1:length(ids)
        rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
        VOCannotations(i).id = [ids(i)];
        VOCannotations(i).bboxes = {rec.objects(:).bbox};
        VOCannotations(i).classes = {rec.objects(:).class};
        VOCannotations(i).difficult = [rec.objects(:).difficult];
    end
    
    save('VOCannotations.mat', 'VOCannotations');
end

fprintf('number of iterations %d\n', length(VOCannotations));

posAnnotations = struct(VOCannotations(1));
negAnnotations = struct(VOCannotations(1));
for i=1:length(VOCannotations)
    %Seperate annotations based on class;
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,VOCannotations(i).classes,'exact');
    diff=[VOCannotations(i).difficult(clsinds)];
    
    % assign ground truth class to image
    if isempty(clsinds)
        % no objects of class
        negAnnotations(end + 1) = VOCannotations(i);
    elseif any(~diff)
        % at least one non-difficult object of class
        posAnnotations(end + 1) = VOCannotations(i);
    %else
        % only difficult objects
        %Note: can add this to pos, but for now choose not to.
    end
end
%Clear out starter annotation
posAnnotations = posAnnotations(2:end);
negAnnotations = negAnnotations(2:end);

%Pass into functions
VOCopts.posAnnotations = posAnnotations;
VOCopts.negAnnotations = negAnnotations;

detector= traina(VOCopts,cls);                            % train detector
fprintf('\n\n\n\n')
fprintf('*************************************\n')
fprintf('         Entering Testing            \n')
fprintf('*************************************\n')
test(VOCopts,cls,detector);                             % test detector
[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);  % compute and display PR #which means precision recall
drawnow;
end


function [newexamples, newgt,newimagenumberlabels,labels] = fillPosExamples(VOCopts, cls, labels)
POS_TRAIN_IMAGES = VOCopts.POS_TRAIN_IMAGES;
MAX_POS_EXAMPLES = VOCopts.MAX_POS_EXAMPLES;
detector.FD=[];
detector.bbox={};
detector.gt=[];
detector.imagenumberlabels = [];
examples = [];
bbIndices = [];

detector.FD = NaN * ones(POS_TRAIN_IMAGES*10,4*VOCopts.numgradientdirections*VOCopts.firstdim* ...
    VOCopts.seconddim);

curAnnotations = VOCopts.posAnnotations(randperm(length(VOCopts.posAnnotations)));
tic;
for i=1:min(POS_TRAIN_IMAGES, length(curAnnotations))
    if (length(detector.gt) > MAX_POS_EXAMPLES) break;
    end
    
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,length(detector.gt),MAX_POS_EXAMPLES);
        drawnow;
        tic;
    end
    
    id = curAnnotations(i).id;
    fd = getFeatures(VOCopts, id{:});
    
    % extract bounding boxes for non-difficult objects
    clsinds=strmatch(cls,curAnnotations(i).classes,'exact');
    diff=[curAnnotations(i).difficult(clsinds)];
    detector.bbox{end+1}=cat(1,curAnnotations(i).bboxes{clsinds(~diff)})';
    a= detector.bbox(end);
    
    %detector.FD = [detector.FD;extractExample(VOCopts, a{1},fd )];
    
    [examples bbIndices] = extractExample(VOCopts, a{1},fd, imread(sprintf(VOCopts.imgpath,id{:})));
    if (size(examples,1) > 0)
        %Add flip of examples
        flipExamples = reshape(examples, [size(examples,1) VOCopts.firstdim VOCopts.seconddim VOCopts.blocksize^2*VOCopts.numgradientdirections]);
        flipExamples = flipExamples(:,:,end:-1:1,:);
        flipExamples = reshape(flipExamples, [size(examples,1) VOCopts.firstdim*VOCopts.seconddim*VOCopts.blocksize^2*VOCopts.numgradientdirections]);
        examples = [examples; flipExamples];
        bbIndices = [bbIndices bbIndices];
        
        for image=1:size(examples,1),
            key= num2str(examples(image,:));
            val = num2str(1);
            labels(key)=val;
        end
        
        detector.FD(length(detector.gt)+1:length(detector.gt)+size(examples,1),:) ...
            = examples;
        %one example for each bounding box,
        %should be a vector of size
        %w*h * 4*9
        labelvalues = bbIndices;
        
        detector.imagenumberlabels = [detector.imagenumberlabels, i*ones(1,size(examples,1))];
        detector.gt = [detector.gt, labelvalues];
    end
end

newgt=detector.gt;
newexamples = detector.FD(1:length(newgt), :);
newimagenumberlabels=detector.imagenumberlabels;
end

function [newexamples, newgt,newimagenumberlabels,labels] = fillNegExamples(VOCopts, cls, ...
    originalexamples, originalgt, originalimagenumbers, labels)

NEG_TRAIN_IMAGES = VOCopts.NEG_TRAIN_IMAGES;
MAX_NEG_EXAMPLES = VOCopts.MAX_NEG_EXAMPLES;
detector.FD=[];
detector.bbox={};
detector.gt=[];
detector.imagenumberlabels = [];
examples = [];
bbIndices = [];

detector.FD = NaN * ones(NEG_TRAIN_IMAGES*10,4*VOCopts.numgradientdirections*VOCopts.firstdim* ...
    VOCopts.seconddim);

if length(originalgt)>0,
    detector.imagenumberlabels=originalimagenumbers;
    detector.gt = originalgt;
    detector.FD(1:length(detector.gt), :) = originalexamples;
end

curAnnotations = VOCopts.negAnnotations(randperm(length(VOCopts.negAnnotations)));
tic;
for i=1:min(NEG_TRAIN_IMAGES, length(curAnnotations))
    if (length(detector.gt) > MAX_NEG_EXAMPLES) break
    end
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,length(detector.gt),MAX_NEG_EXAMPLES);
        drawnow;
        tic;
    end
    
    
    id = curAnnotations(i).id;
    fd = getFeatures(VOCopts, id{:});
    
    % extract bounding boxes for non-difficult objects
    clsinds=strmatch(cls,curAnnotations(i).classes,'exact');
    diff=[curAnnotations(i).difficult(clsinds)];
    detector.bbox{end+1}=cat(1,curAnnotations(i).bboxes{clsinds(~diff)})';
    a= detector.bbox(end);
    
    %detector.FD = [detector.FD;extractExample(VOCopts, a{1},fd )];
    
    [examples bbIndices] = extractExample(VOCopts, a{1},fd, imread(sprintf(VOCopts.imgpath,id{:})));
    if (size(examples,1) > 0)
        %Add flip of examples
        flipExamples = reshape(examples, [size(examples,1) VOCopts.firstdim VOCopts.seconddim VOCopts.blocksize^2*VOCopts.numgradientdirections]);
        flipExamples = flipExamples(:,:,end:-1:1,:);
        flipExamples = reshape(flipExamples, [size(examples,1) VOCopts.firstdim*VOCopts.seconddim*VOCopts.blocksize^2*VOCopts.numgradientdirections]);
        examples = [examples; flipExamples];
        bbIndices = [bbIndices bbIndices];
        
        for image=1:size(examples,1),
            key= num2str(examples(image,:));
            val = num2str(1);
            labels(key)=val;
        end
        
        detector.FD(length(detector.gt)+1:length(detector.gt)+size(examples,1),:) ...
            = examples;
        %one example for each bounding box,
        %should be a vector of size
        %w*h * 4*9
        labelvalues = -1*ones(1,size(examples,1));
        
        detector.imagenumberlabels = [detector.imagenumberlabels, i*ones(1,size(examples,1))];
        detector.gt = [detector.gt, labelvalues];
    end
end

newgt=detector.gt;
newexamples = detector.FD(1:length(newgt), :);
newimagenumberlabels=detector.imagenumberlabels;
end

%Fill examples to get a full set for training
function [newexamples, newgt,newimagenumberlabels,labels] = fillexamples(VOCopts,cls, ...
    originalexamples, originalgt, originalimagenumbers,labels)

[posexamples, posgt, posimagenumberlabels, labels] = fillPosExamples(VOCopts,cls,labels);
[negexamples, neggt, negimagenumberlabels, labels] = fillNegExamples(VOCopts,cls, ...
    originalexamples, originalgt, originalimagenumbers,labels);

newgt=[posgt neggt];
newexamples = [posexamples; negexamples];
newimagenumberlabels=[posimagenumberlabels negimagenumberlabels];
end


function sanitycheck(labels, savedfeatures, savedgt)

for i=1:length(savedgt)
    if str2num(labels(num2str(savedfeatures(i,:))))-savedgt~=0
        while 1,
            fprintf('pretty fundamental issue');
        end
    end
end
end

function [savedfeatures, savedgt, savedimagelabels]=extractHardExamples(VOCopts, detector, newexamples,newgt,newimagelabels)
HARDNESS_CUTOFF = VOCopts.HARDNESS_CUTOFF;

binaryizegt = 2*((newgt>0)-.5);

scores = detector.multiplier*[newexamples,ones(size(newexamples,1),1)] * detector.w';

negScores = scores.*(binaryizegt < 0)';
hardExamples = (negScores > HARDNESS_CUTOFF).*(binaryizegt < 0)';
hardIndices = find(hardExamples);

savedfeatures = newexamples(hardIndices,:);

savedgt = newgt(hardIndices);
savedimagelabels = newimagelabels(hardIndices);



%testgt' - 2*((scores > 0 )-.5)


%disp 'how bad we did on the worst'
%[predicted_label, accuracy] ...
% = liblinearpredict(savedgt',sparse(savedfeatures),svmStruct);
fprintf('number of saved examples: %d\n',size(savedfeatures,1));

end


function [detector] = traina(VOCopts,cls)

labels = containers.Map();

savedfeatures = [];
savedgt = [];
savedimagelabel=[];

for i=1:VOCopts.rootfilterminingiters, %this is finding "Root Filter Initialization"
    fprintf('we are on iteration %d\n', i);
    [newexamples, newgt,newimagenumbers,labels] = ...
        fillexamples(VOCopts, cls, savedfeatures, savedgt, savedimagelabel,labels);
    %sanitycheck(labels, newexamples,newgt);
    fprintf('number of examples to train on: %d\n',length(newgt));
    
    %Permute examples
    perm = randperm(length(newgt));   
    newgt = newgt(perm);
    newexamples = newexamples(perm, :);
    newimagenumbers = newimagenumbers(perm);
    
    %Train detector
    binaryizegt = 2*((newgt>0)-.5);
    detector = detectorTrain(binaryizegt,newexamples);

    %Output some performance stats
    disp 'How we do overall'
    naiveperformance = abs(sum(binaryizegt)/length(binaryizegt))/2 + .5 %baseline
    binaryizegt = 2*((newgt>0)-.5);
    [predicted_label, accuracy] ...
        = liblinearpredict(binaryizegt',sparse(newexamples),detector);
    scores = detector.multiplier*[newexamples,ones(size(newexamples,1),1)] * detector.w';
    empiricalerrors= sum(abs(2*((scores > 0 )-.5) - predicted_label))/2;
    fprintf('number of errors %d vs. number of examples %d', ...
        empiricalerrors,length(scores));

    %Mine hard examples
    [savedfeatures, savedgt, savedimagelabel] = extractHardExamples(VOCopts, detector, ...
        newexamples, newgt, newimagenumbers);
end

disp 'training latently'
for i=1:VOCopts.rootfilterupdateiters,%this step is "Root Filter Update"
    [newexamples, newgt,newimagenumbers,labels] = ...
        fillexamples(VOCopts, cls, savedfeatures, savedgt, savedimagelabel,labels);
    fprintf('Training latently iteration %d with this many examples %d\n', i, length(newgt));
    tic;
    [newexamples] = findNewPositives(VOCopts, cls, newgt, newexamples, newimagenumbers,detector);
    toc
    [newdetector, savedfeatures, savedgt, savedimagelabel] = extractHardExamples(newexamples, newgt,newimagenumbers);
    %detector = detectorTrain(newgt, newexamples);
end

tic;
[partFilters partBBoxes] = initializePartsFromRoot(VOCopts, detector.w);
toc

try
    save('detector.mat','detector');
catch
    disp 'Sorry, couldnt save detector'
end

[detector]=finaltest(VOCopts, cls, labels, detector);
end


function [detector] = finaltest(VOCopts, cls, labels, detector)
disp 'final showdown'

[newtestexamples, newtestgt] = fillexamples(VOCopts, cls, [], [], [],labels);

binaryizegt = newtestgt>0;
disp 'testing detector'
naiveperformance = abs((sum(binaryizegt>0) - sum(binaryizegt<0))/length(binaryizegt))/2 + .5 %baseline

scores = [newtestexamples,ones(size(newtestexamples,1),1)] * detector.w';

binaryizegt = 2*((newtestgt>0)-.5);
[predicted_label_test, accuracy] ...
    = liblinearpredict(binaryizegt',sparse(newtestexamples),detector);

incorrectpredictions=sum(abs(2*((detector.multiplier*scores > 0 )-.5) ...
    - predicted_label_test));
fprintf('this really should be zero %d\n', incorrectpredictions);


%svmStruct = svmtrain(detector.gt',detector.FD);
%[predicted_label, accuracy, decision_values] ...
% = svmpredict(detector.gt',detector.FD,svmStruct);

%correct = 0;
%for i = 1:size(detector.gt,2),
% label = svmpredict(detector.gt(i),detector.FD(i,:),svmStruct);
% %label = svmpredict(svmStruct, detector.FD(i,:));
% if abs(label - detector.gt(i))<1e-3,
% correct = correct +1;
% end
%end

end
% run detector on test images
%TEST_IMAGES=length(ids);
function out = test(VOCopts,cls,detector)
% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

TEST_IMAGES=VOCopts.TEST_IMAGES;
%TEST_IMAGES=length(ids);

fprintf('number of images we could hope to use %d \n', length(ids));



% create results file
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% apply detector to each image
tic;
%for i=1:length(ids)
for i=1:TEST_IMAGES,
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,TEST_IMAGES);
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        disp 'failed to load features'
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        try
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            
        end
    end
    
    % compute confidence of positive classification and bounding boxes
    I = imread(sprintf(VOCopts.imgpath,ids{i}));
   
    [c,BB]= detect(VOCopts,detector,fd,I,i);
    
    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %d %d %d %d\n',ids{i},c(j),BB(:,j));
    end
end

% close results file
fclose(fid);

end



% trivial detector: confidence is computed as in example_classifier, and
% bounding boxes of nearest positive training image are output
function [c,BB] = detect(VOCopts,detector,fd,I,number)

c = [];
BB = [];

bbox = [1; 1; size(I,2); size(I,1)];
%[c, BB, ~] = findBestNegativeExample(VOCopts, fd, detector, [],bbox,VOCopts.rootsamplestesting);

for pyramidIndex=1:length(fd)
    currlevel = fd{pyramidIndex};
    xdim = size(currlevel,1);
    ydim = size(currlevel,2);
    for x = 1+VOCopts.firstdim/2 :xdim - VOCopts.firstdim/2,
        for y = 1+VOCopts.seconddim/2:ydim - VOCopts.seconddim/2,
            [pixelBox, pixelCenter]=HOGSpaceToPixelSpace(VOCopts, [x;y],pyramidIndex);
            [~, HOGVector] = pixelSpaceToHOGSpace(VOCopts, fd, pixelCenter,pyramidIndex);
            score = detector.multiplier*[HOGVector,1]*detector.w';
            if score>0,
                c = [c score];
                BB = [BB pixelBox];
                %disp 'gotta match'
            end
        end
    end
end

[c,BB] = nonMaximalSupression(c,BB);


disp 'about to draw'
drawWithBB(I,BB,sprintf('image%d.png', number));



fprintf('number of matches found %d in image %d\n', length(c),number);

end

function [newc, newBB] = nonMaximalSupression(c,BB)
newc = [];
newBB = [zeros(size(BB))];

while max(c)>1,
    [val, index] = max(c);
    newc(end+1) =c(index);
    newBB(:, length(newc)) = BB(:, index);
    
    c(index)=-1;
    for bbindex = 1:length(c),
        if calcMinOverlap(BB(:, bbindex), BB(:, index)),
            c(bbindex)=-1;
        end
    end
end

if ~isempty(newc)
    newBB = newBB(:, 1:size(newc,2));
end

end
