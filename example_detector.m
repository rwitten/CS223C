function example_detector

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

VOCopts.blocksize=2;
VOCopts.cellsize =  8;
VOCopts.numgradientdirections = 9;
VOCopts.firstdim = 10;
VOCopts.seconddim=6;
VOCopts.rootfilterminingiters=1;
VOCopts.rootfilterupdateiters=1;
VOCopts.TRAIN_IMAGES=50; %this is the size of cache in terms of number of images
VOCopts.pyramidscale = 1/1.1;
VOCopts.hognormclip = 1;
%VOCopts.firstdim = 32; %empirical average!
%VOCopts.seconddim=22;  %empirical average!

% train and test detector for each class
cls='person';
detector=train(VOCopts,cls);                            % train detector
fprintf('\n\n\n\n')
fprintf('*************************************\n')
fprintf('         Entering Testing            \n')
fprintf('*************************************\n')
test(VOCopts,cls,detector);                             % test detector
 detector=train(VOCopts,cls);                            % train detector
%test(VOCopts,cls,detector);                             % test detector
%[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);  % compute and display PR #which means precision recall
drawnow;



% train detector
function [newexamples, newgt,newimagenumberlabels,labels] = fillexamples(VOCopts,cls, ...
    originalexamples, originalgt, originalimagenumbers,labels)

% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

TOTAL_IMAGES=length(ids);
%TOTAL_IMAGES=2500;
%TOTAL_IMAGES=100;
TRAIN_IMAGES=VOCopts.TRAIN_IMAGES;

% extract features and bounding boxes
detector.FD=[];
detector.bbox={};
detector.gt=[];
detector.imagenumberlabels = []; 
tic;
examples = [];

detector.FD = NaN * ones(TRAIN_IMAGES,4*VOCopts.numgradientdirections*VOCopts.firstdim* ...
    VOCopts.seconddim);

if length(originalgt)>0,
    detector.imagenumberlabels=originalimagenumbers;
    detector.gt = originalgt;
    detector.FD(1:min(TRAIN_IMAGES,length(detector.gt)), :) = originalexamples;
end

while TRAIN_IMAGES>length(detector.gt),
    i = floor(rand*TOTAL_IMAGES)+1;
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,length(detector.gt),TRAIN_IMAGES);
        drawnow;
        tic;
    end
    % read annotation
    
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    diff=[rec.objects(clsinds).difficult];
    
    % assign ground truth class to image
    if isempty(clsinds)
        gt=-1;          % no objects of class
    elseif any(~diff)
        gt=1;           % at least one non-difficult object of class
    else
        gt=0;           % only difficult objects
    end
    if gt
        % extract features for image
        fd = getFeatures(VOCopts, ids{i});
        
        %VOCopts.exfdpath
        
        %detector.FD(1:length(fd),end+1)=fd;
        
        % extract bounding boxes for non-difficult objects
        
        detector.bbox{end+1}=cat(1,rec.objects(clsinds(~diff)).bbox)';
        a= detector.bbox(end);
        
        %detector.FD = [detector.FD;extractExample(VOCopts, a{1},fd )]; 

        examples = extractExample(VOCopts, a{1},fd);
        if (size(examples,1) > 0)
            for image=1:size(examples,1),
                key= num2str(examples(image,:));
                val = num2str(gt);
                labels(key)=val;
            end

            detector.FD(length(detector.gt)+1:length(detector.gt)+size(examples,1),:) ...
                = examples;
                                                       %one example for each bounding box, 
                                                       %should be a vector of size 
                                                       %w*h * 4*9
            if gt==-1,
              labelvalues = gt*ones(1,size(examples,1));
            else
              labelvalues = 1:size(examples);
            end
            detector.imagenumberlabels = [detector.imagenumberlabels, i*ones(1,size(examples,1))]; 
            detector.gt = [detector.gt, labelvalues]; 
        end
        
    end
end
newgt=detector.gt;
newexamples = detector.FD(1:length(newgt), :);
newimagenumberlabels=detector.imagenumberlabels;

function sanitycheck(labels, savedfeatures, savedgt)

for i=1:length(savedgt)
    if str2num(labels(num2str(savedfeatures(i,:))))-savedgt~=0
        while 1,
          fprintf('pretty fundamental issue');
        end
    end
end

function [newdetector,hardexamples, hardgt, hardimagelabel]=extractHardExamples(newexamples,newgt,newimagelabels)
binaryizegt = 2*((newgt>0)-.5);
svmStruct = liblineartrain(binaryizegt',sparse(newexamples),'-s 2 -B 1 -q');

%[predicted_label, accuracy, decision_values] ...
%   = svmpredict(traingt',trainfd,svmStruct);


disp 'How we do overall'
naiveperformance = abs(sum(binaryizegt)/length(binaryizegt))/2 + .5 %baseline
binaryizegt = 2*((newgt>0)-.5);
[predicted_label, accuracy] ...
   = liblinearpredict(binaryizegt',sparse(newexamples),svmStruct);


scores = [newexamples,ones(size(newexamples,1),1)] * svmStruct.w';

empiricalerrors= sum(2*((scores > 0 )-.5) == predicted_label);
                                                    % this should be 0
svmStruct.multiplier = -1;
if empiricalerrors>0,
    fprintf('number of errors %d vs. number of examples %d', ...
        empiricalerrors,length(scores));
    svmStruct.multiplier = 1;
end
howbadwedid = svmStruct.multiplier .*scores.*newgt';

badnesscutoff = median(howbadwedid)

savedfeatures = NaN * ones(length(howbadwedid),...
    size(newexamples,2));

savedgt = [];
savedimagelabel = [];

for i=1:length(howbadwedid),
    if howbadwedid(i)<badnesscutoff,
        savedgt(end+1) = newgt(i);
        savedimagelabel(end+1) = newimagelabels(i);
        savedfeatures(length(savedgt),:) = newexamples(i,:)';
    end
end
newdetector=svmStruct;
hardexamples=savedfeatures(1:length(savedgt),:);
hardgt = savedgt;
hardimagelabel = savedimagelabel;

fprintf('number of saved examples: %d\n',size(savedfeatures,1));

%testgt' - 2*((scores > 0 )-.5)


%disp 'how bad we did on the worst'
%[predicted_label, accuracy] ...
%   = liblinearpredict(savedgt',sparse(savedfeatures),svmStruct);


function [detector] = train(VOCopts,cls)

labels = containers.Map();

savedfeatures = [];
savedgt = [];
savedimagelabel=[];

for i=1:VOCopts.rootfilterminingiters, %this is finding "Root Filter Initialization"
    fprintf('we are on iteration %d\n', i);
    [newexamples, newgt,newimagenumbers,labels] = ...
        fillexamples(VOCopts, cls, savedfeatures, savedgt, savedimagelabel,labels);
    sanitycheck(labels, newexamples,newgt);
    fprintf('number of examples to train on: %d\n',length(newgt));
    
    perm = randperm(length(newgt));
    
    newgt = newgt(perm);
    newexamples = newexamples(perm, :);
    newimagenumbers = newimagenumbers(perm);
    
    [newdetector, savedfeatures, savedgt, savedimagelabel] = extractHardExamples(newexamples, newgt,newimagenumbers);
end

detector=newdetector;

for i=1:VOCopts.rootfilterupdateiters,%this step is "Root Filter Update"
    [newexamples] = findNewPositives(VOCopts, cls, newgt, newexamples, newimagenumbers,detector);
    size(newexamples)
    binaryizegt = 2*((newgt>0)-.5);
    detector = liblineartrain(binaryizegt', sparse(newexamples), '-s 2 -B 1 -q');
end

[detector]=finaltrainandtest(VOCopts, cls, newgt, newexamples,labels);


function [detector] = finaltrainandtest(VOCopts, cls, newgt, newexamples,labels)

disp 'final showdown'
binaryizegt = 2*((newgt>0)-.5);
detector = liblineartrain(binaryizegt',sparse(newexamples), '-s 2 -B 1 -q');
                      %we need to train a final detector on hard examples
disp 'detector trained'                   
[newtestexamples, newtestgt] = fillexamples(VOCopts, cls, [], [], [],labels);


disp 'testing detector'
naiveperformance = abs((sum(newtestgt>0) - sum(newtestgt<0))/length(newtestgt))/2 + .5 %baseline

scores = [newtestexamples,ones(size(newtestexamples,1),1)] * detector.w';

binaryizegt = 2*((newtestgt>0)-.5);
[predicted_label_test, accuracy] ...
       = liblinearpredict(binaryizegt',sparse(newtestexamples),detector);
   
   
 if sum(abs(2*((scores > 0 )-.5) - predicted_label_test)) < 1,
     detector.multiplier = 1;
 else
     detector.multiplier = -1;
 end
 
 incorrectpredictions=sum(abs(2*((detector.multiplier*scores > 0 )-.5) ...
     - predicted_label_test));
 fprintf('this really should be zero %d\n', incorrectpredictions);


%svmStruct = svmtrain(detector.gt',detector.FD);
%[predicted_label, accuracy, decision_values] ...
% = svmpredict(detector.gt',detector.FD,svmStruct);

%correct = 0;
%for i = 1:size(detector.gt,2),
%    label = svmpredict(detector.gt(i),detector.FD(i,:),svmStruct);
%    %label = svmpredict(svmStruct, detector.FD(i,:));
%    if abs(label - detector.gt(i))<1e-3,
%        correct = correct +1;
%    end
%end

% run detector on test images
function out = test(VOCopts,cls,detector) 
%TEST_IMAGES=length(ids);
TEST_IMAGES=100;

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

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
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
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


% trivial detector: confidence is computed as in example_classifier, and
% bounding boxes of nearest positive training image are output
function [c,BB] = detect(VOCopts,detector,fd,I,number)

c = [];
BB = [];

<<<<<<< HEAD
for pyramidIndex=1:length(fd)
    currlevel = fd{pyramidIndex};
    xdim = size(currlevel,1);
    ydim = size(currlevel,2);
    for x = 1+VOCopts.firstdim/2 :xdim - VOCopts.firstdim/2,
        for y = 1+VOCopts.seconddim/2:ydim - VOCopts.seconddim/2,
            [pixelBox, pixelCenter]=HOGSpaceToPixelSpace(VOCopts, [x;y],pyramidIndex);
            [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, fd, pixelCenter,pyramidIndex);
            score = detector.multiplier*[HOGVector,1]*detector.w';
=======
for i=1:length(fd)
    xdim = size(fd{i},1);
    ydim = size(fd{i},2);
    for x = 1+VOCopts.firstdim/2 :xdim - VOCopts.firstdim/2,
        for y = 1+VOCopts.seconddim/2:ydim - VOCopts.seconddim/2,
            [pixelBox, pixelCenter]=HOGSpaceToPixelSpace(VOCopts, [x;y], i);
            [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, fd{i}, pixelCenter);
            score = detector.multiplier*HOGVector*detector.w';
>>>>>>> 0c6f84d353ed8ed2ccda5781fde8a62b08062bb0
            if score>1,
                c = [c score];
                newboundingbox = [pixelBox(3); pixelBox(1);pixelBox(4); pixelBox(2)];
                BB = [BB newboundingbox];
                %disp 'gotta match'
            end

        end
    end
end

for k = 1:size(BB,2)
    pixelBox=BB(:,k);
    for x=1:size(I,1),
        for y=1:size(I,2),
            if x > pixelBox(1) && x < pixelBox(3) && y>pixelBox(2) && y<pixelBox(4)
                I(x,y,1) = 1e4;
            end
        end
    end
end

imwrite(I, sprintf('image%d.png', number), 'png');


fprintf('number of matches found %d\n', length(c));


% iterate through and fire if we're bigger than some cutoff.
