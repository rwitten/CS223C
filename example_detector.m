function example_detector

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% train and test detector for each class
cls='person';
detector=train(VOCopts,cls);                            % train detector
test(VOCopts,cls,detector);                             % test detector
%[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);  % compute and display PR #which means precision recall
drawnow;



% train detector
function detector = train(VOCopts,cls)
TRAIN_IMAGES=2500;

VOCopts.blocksize=2;
VOCopts.cellsize =  8;
VOCopts.numgradientdirections = 9;
VOCopts.firstdim = 10; %empirical average!
VOCopts.seconddim=6;  %empirical average!
%VOCopts.firstdim = 32; %empirical average!
%VOCopts.seconddim=22;  %empirical average!




% load 'train' image set
ids=textread(sprintf(VOCopts.imgsetpath,'train'),'%s');

% extract features and bounding boxes
detector.FD=[];
detector.bbox={};
detector.gt=[];
tic;
examples = [];
for i=1:TRAIN_IMAGES,
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
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
        
        %VOCopts.exfdpath
        
        %detector.FD(1:length(fd),end+1)=fd;
        
        % extract bounding boxes for non-difficult objects
        
        detector.bbox{end+1}=cat(1,rec.objects(clsinds(~diff)).bbox)';
        a= detector.bbox(end);
        HOGFeatures = extractExample(VOCopts, a{1},fd ); 
                                                     %one example from each image, 
                                                     %should be a vector of size 
                                                     %w*h * 4*9
       if size(HOGFeatures,1)>0,  %if we managed to extract a bounding box                                                   
          detector.FD(end+1,1:length(HOGFeatures))=HOGFeatures;%data
          detector.gt(end+1)=gt;% mark image as positive or negative
       end
    end
end    

svmStruct = svmtrain(detector.gt',detector.FD);
[predicted_label, accuracy, decision_values] = svmpredict(detector.gt',detector.FD,svmStruct);

sum(abs(predicted_label- detector.gt')<1e-3)/size(detector.gt',1)

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
return
TEST_IMAGES=100;

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% apply detector to each image
tic;
for i=1:TEST_IMAGES
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

% compute confidence
d=sum(fd.*fd)+sum(detector.FD.*detector.FD)-2*fd'*detector.FD;
dp=min(d(detector.gt>0));
dn=min(d(detector.gt<0));
c=dn/(dp+eps);

% copy bounding boxes from nearest positive image
pinds=find(detector.gt>0);
[dp,di]=min(d(pinds));
pind=pinds(di);
BB=detector.bbox{pind};

% replicate confidence for each detection
c=ones(size(BB,2),1)*c;