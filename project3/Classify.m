function [] = Classify()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Initialize parameters of SPM classification
params = initParams();

%Build pyramids for each class
totalPyramids = 0;
pyramid_cells = cell(params.num_classes,1);
for i=1:params.num_classes
    %Build list of filepaths
    cur_image_dir = strcat(params.image_dir, '/',params.class_names{i});
    cur_data_dir = strcat(params.data_dir, '/',params.class_names{i});
    fnames = dir(fullfile(cur_image_dir, '*.jpg'));
    num_files = size(fnames,1);
    filenames = cell(num_files,1);
    for f = 1:num_files
        filenames{f} = fnames(f).name;
    end
    
    %Run SPM
    pyramid_cells{i} = BuildPyramid(filenames, cur_image_dir, cur_data_dir, ...
        params.max_image_size, params.dictionary_size, params.num_texton_images, ...
        params.pyramid_levels, params.can_skip);
    
    totalPyramids = totalPyramids + size(pyramid_cells{i},1);
end

%Separate into training and testing datasets
train_pyramids = zeros(ceil(1.05*params.percent_train*totalPyramids), size(pyramid_cells{1},2));
train_labels = zeros(ceil(1.05*params.percent_train*totalPyramids), 1);
test_pyramids = zeros(ceil(1.05*(1-params.percent_train)*totalPyramids), size(pyramid_cells{1},2));
test_labels = zeros(ceil(1.05*(1-params.percent_train)*totalPyramids), 1);
curTrainEnd = 0;
curTestEnd = 0;
for i=1:length(pyramid_cells)
    cur_pyramids = squeeze(pyramid_cells{i});
    perm = randperm(size(cur_pyramids,1));
    trainperm = perm(1:floor(params.percent_train*end));
    testperm = perm(ceil(params.percent_train*end):end);
    train_pyramids(curTrainEnd + (1:length(trainperm)),:) = cur_pyramids(trainperm,:);
    test_pyramids(curTestEnd + (1:length(testperm)),:) = cur_pyramids(testperm,:);
    train_labels(curTrainEnd + (1:length(trainperm))) = i;
    test_labels(curTestEnd + (1:length(testperm))) = i;
    curTrainEnd = curTrainEnd + length(trainperm);
    curTestEnd = curTestEnd + length(testperm);
end
train_pyramids = train_pyramids(1:curTrainEnd,:);
test_pyramids = test_pyramids(1:curTestEnd,:);
train_labels = train_labels(1:curTrainEnd);
test_labels = test_labels(1:curTestEnd);

%Train detector
model = train(train_labels,sparse(train_pyramids));

%Test detector
[~, accuracy] = predict(test_labels, sparse(test_pyramids), model);
accuracy

end

function params = initParams()
    load('class_names.mat', 'classes');

    params.image_dir = 'images'; 
    params.data_dir = 'data';
    params.class_names = classes;
    params.num_classes = length(params.class_names);
    params.max_image_size = 1000;
    params.dictionary_size = 200;
    params.num_texton_images = 50;
    params.pyramid_levels = 4;
    params.can_skip = 1;
    params.percent_train = 0.7;
end