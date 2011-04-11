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
VOCopts.miningiters=5;
%VOCopts.firstdim = 32; %empirical average!
%VOCopts.seconddim=22;  %empirical average!

% train and test detector for each class
cls='person';
detector=train(VOCopts,cls);                            % train detector
%test(VOCopts,cls,detector);                             % test detector
%[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);  % compute and display PR #which means precision recall
drawnow;



% train detector
function [newexamples, newgt,labels] = fillexamples(VOCopts,cls, originalexamples, originalgt, labels)

% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

TOTAL_IMAGES=length(ids);
TRAIN_IMAGES=100;

% extract features and bounding boxes
detector.FD=[];
detector.bbox={};
detector.gt=[];
tic;
examples = [];

detector.FD = NaN * ones(TRAIN_IMAGES*10,4*VOCopts.numgradientdirections*VOCopts.firstdim* ...
    VOCopts.seconddim);

if length(originalgt)>0,
    detector.gt = originalgt;
    detector.FD(1:length(detector.gt), :) = originalexamples;
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
        try
            % try to load features
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            % compute and save features
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(VOCopts,I);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        imwrite(I, sprintf('savedimage%dwithagt%d.png', i,gt), 'png');
        
        %VOCopts.exfdpath
        
        %detector.FD(1:length(fd),end+1)=fd;
        
        % extract bounding boxes for non-difficult objects
        
        detector.bbox{end+1}=cat(1,rec.objects(clsinds(~diff)).bbox)';
        a= detector.bbox(end);
        
        %detector.FD = [detector.FD;extractExample(VOCopts, a{1},fd )]; 
        examples = extractExample(VOCopts, a{1},fd );
        
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
        
        detector.gt = [detector.gt, gt*ones(1,size(examples,1))]; 
        
    end
end
newgt=detector.gt;
newexamples = detector.FD(1:length(newgt), :);

function sanitycheck(labels, savedfeatures, savedgt)

for i=1:length(savedgt)
    if str2num(labels(num2str(savedfeatures(i,:))))-savedgt~=0
        while 1,
          fprintf('pretty fundamental issue');
        end
    end
end

function [newdetector,hardexamples, hardgt]=extractHardExamples(newexamples,newgt)
halfway = floor(length(newgt)/2);

traingt = newgt(1:halfway);
testgt = newgt(halfway+1:end);
trainfd = newexamples(1:halfway, :);
testfd = newexamples(halfway+1:end, :);

%svmStruct = svmtrain(traingt',trainfd);
svmStructTrain = liblineartrain(traingt',sparse(trainfd), '-s 2 -B 1 -q');
svmStructTest = liblineartrain(testgt',sparse(testfd), '-s 2 -B 1 -q'); 
                                        %svmStructTrain is trained on
                                        %train data and svmStructTest
                                        %is trained on test data.

%[predicted_label, accuracy, decision_values] ...
%   = svmpredict(traingt',trainfd,svmStruct);


disp 'How we do overall'
[predicted_label_train, accuracy] ...
   = liblinearpredict(testgt',sparse(testfd),svmStructTrain);
[predicted_label_test, accuracy] ...
   = liblinearpredict(traingt',sparse(trainfd),svmStructTest);
naiveperformance = abs(sum(newgt)/length(newgt))/2 + .5 %baseline
size(testfd)
scorestrain = [testfd,ones(size(testfd,1),1)] * svmStructTrain.w';
scorestest = [trainfd,ones(size(trainfd,1),1)] * svmStructTest.w';

howbadwedidtrain = scorestrain.*testgt';
howbadwedidtest = scorestest.*traingt';

empiricalerrorstrain = sum(2*((scorestrain > 0 )-.5) == predicted_label_train);
                                                    % this should be 0
empiricalerrorstest = sum(2*((scorestest > 0 )-.5) == predicted_label_test);
                                                    % this should be 0

if empiricalerrorstest>0,
    fprintf('number of errors %d vs. number of examples %d', ...
        empiricalerrorstest,length(scorestest));
    howbadwedidtest = howbadwedidtest*(-1);
end

if empiricalerrorstrain>0,
    fprintf('number of errors %d vs. number of examples %d', ...
        empiricalerrorstrain,length(scorestest));
    howbadwedidtrain = howbadwedidtrain*(-1);
end

badnesscutofftrain = median(howbadwedidtrain)
badnesscutofftest = median(howbadwedidtest)

savedfeatures = NaN * ones(length(howbadwedidtrain)+length(howbadwedidtest),...
    size(newexamples,2));

savedgt = [];

for i=1:length(howbadwedidtrain),
    if howbadwedidtrain(i)<badnesscutofftrain,
        savedgt(end+1) = testgt(i);
        savedfeatures(length(savedgt),:) = testfd(i,:)';
    end
end

for i=1:length(howbadwedidtest),
    if howbadwedidtest(i)<badnesscutofftest,
        savedgt(end+1) = traingt(i);
        savedfeatures(length(savedgt),:) = trainfd(i,:)';
    end
end

savedfeatures= savedfeatures(1:length(savedgt),:);

fprintf('number of saved examples: %d\n',size(savedfeatures,1));

%testgt' - 2*((scores > 0 )-.5)


%disp 'how bad we did on the worst'
%[predicted_label, accuracy] ...
%   = liblinearpredict(savedgt',sparse(savedfeatures),svmStruct);


function [detector] = train(VOCopts,cls)

labels = containers.Map();

savedfeatures = [];
savedgt = [];

for i=1:VOCopts.miningiters,
    fprintf('we are on iteration %d\n', i);
    [newexamples, newgt,labels] = fillexamples(VOCopts, cls, savedfeatures, savedgt,labels);
    sanitycheck(labels, newexamples,newgt);
    fprintf('number of examples to train on: %d\n',length(newgt));
    
    perm = randperm(length(newgt));
    
    newgt = newgt(perm);
    newexamples = newexamples(perm, :);
    
    [newdetector, hardexamples, hardgt] = extractHardExamples(newexamples, newgt);
    
    
end

disp 'final showdown'
detector = liblineartrain(newgt',sparse(newexamples), '-s 2 -B 1 -q');
                      %we need to train a final detector on hard examples
disp 'detector trained'                   
[newtestexamples, newtestgt] = fillexamples(VOCopts, cls, [], [], labels);


disp 'testing detector'
naiveperformance = abs(sum(newtestgt)/length(newtestgt))/2 + .5 %baseline

scores = [newtestexamples,ones(size(newtestexamples,1),1)] * detector.w';
[predicted_label_test, accuracy] ...
       = liblinearpredict(newtestgt',sparse(newtestexamples),detector);
   
 if sum(abs(2*((scorestrain > 0 )-.5) - predicted_label_train)) < 1,
     detector.multiplier = 1;
 else
     detector.multiplier = -1;
 end
 
 incorrectpredictions=sum(abs(2*((detector.multiplier*scorestrain > 0 )-.5) ...
     - predicted_label_train));
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
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
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
    [c,BB]=detect(VOCopts,detector,fd);

    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %d %d %d %d\n',ids{i},c(j),BB(:,j));
    end
end

% close results file
fclose(fid);

function fd = extractfd(VOCopts,I)

fd = HOG(VOCopts,I);

% trivial detector: confidence is computed as in example_classifier, and
% bounding boxes of nearest positive training image are output
function [c,BB] = detect(VOCopts,detector,fd)

xdim = size(fd,1);
ydim = size(fd,2);

c = [];
BB = [];

for x = 1+VOCopts.firstdim/2 :xdim - VOCopts.firstdim/2,
    for y = 1+VOCopts.seconddim/2:ydim - VOCopts.seconddim/2,
        [pixelBox, pixelCenter]=HOGSpaceToPixelSpace(VOCopts, [x;y]);
        [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, fd, pixelCenter);
        score = detector.multiplier*HOGVector*detector.w';
        if score>1,
            c = [c score];
            newboundingbox = [pixelBox(3); pixelBox(1);pixelBox(4); pixelBox(2)];
            BB = [BB newboundingbox];
            %disp 'gotta match'
        end
        
    end
end

fprintf('number of matches found %d out of %d\n', length(c), (xdim-VOCopts.firstdim)...
    *(ydim-VOCopts.seconddim));


% iterate through and fire if we're bigger than some cutoff.