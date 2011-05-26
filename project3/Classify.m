function [] = Classify()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Initialize parameters of SPM classification
params = initParams();

%Build pyramids for each class
totalLabels = [];
filenames = {};
for i=1:params.num_classes
    %Build list of filepaths
    cur_image_dir = strcat(params.image_dir, '/',params.class_names{i});
    fnames = dir(fullfile(cur_image_dir, '*.jpg'));
    num_files = size(fnames,1);
    newfilenames = cell(num_files,1);
    for f = 1:num_files
        newfilenames{f} =  strcat(params.class_names{i}, '/', fnames(f).name);
    end
    filenames(end + (1:num_files)) = newfilenames;
    totalLabels = [totalLabels; i*ones(num_files,1)];
end    
filenames = filenames';

%Separate into training and testing datasets
train_filenames = {};
test_filenames = {};
train_labels = [];
test_labels = [];
for i=1:params.num_classes
    cur_filenames = filenames(totalLabels == i);
    perm = randperm(size(cur_filenames,1));
    trainperm = perm(1:floor(params.percent_train*end));
    testperm = perm(ceil(params.percent_train*end):end);
    train_filenames = [train_filenames; cur_filenames(trainperm)];
    test_filenames = [test_filenames; cur_filenames(testperm)];
    train_labels = [train_labels; i*ones(length(trainperm),1)];
    test_labels = [test_labels; i*ones(length(testperm),1)];
end
clear total_labels;
clear filenames;
fprintf('Data separated');


%Note: Codebook technically could be generated from testing data with this
%approach, easy to fix up later though.
train_pyramids = BuildPyramid(train_filenames, params);
test_pyramids = BuildPyramid(test_filenames, params);

size(train_pyramids)
size(test_pyramids)
fprintf('Pyramids built');
clear filenames;


%Apply kernels if you want
if params.apply_kernel
    train_data = hist_isect_c(train_pyramids, train_pyramids);
    test_data = hist_isect_c(test_pyramids, train_pyramids);
    fprintf('kernel applied');
else
    train_data = train_pyramids;
    test_data = test_pyramids;
    fprintf('No kernel applied');
end
clear train_pyramids;
clear test_pyramids;




%Train detector
model = train(train_labels,sparse(train_data));
[~, accuracy] = predict(train_labels, sparse(train_data), model);
accuracy

%Test detector
[~, accuracy] = predict(test_labels, sparse(test_data), model);
accuracy

end

function params = initParams()
    load('class_names.mat', 'classes');
    
    params.do_ppmi = 0;
    if ~params.do_ppmi,
        params.image_dir = 'images'; 
        params.data_dir = 'data';
    else
        params.image_dir = 'ppmi/norm_image/play_instrument'; 
        params.data_dir = 'data_ppmi';
    end
    
    params.class_names = classes;
    params.num_classes = 3;%length(params.class_names);
    params.max_image_size = 1000;
    params.dictionary_size = 200;
    params.num_texton_images = 50;
    params.pyramid_levels = 4;
    params.max_pooling = 1;
    params.do_llc = 0;
    params.apply_kernel = 1;
    params.can_skip = 1;
    params.can_skip_sift = 1;
    params.can_skip_calcdict = 1;
    params.can_skip_buildhist = 0;
    params.can_skip_compilepyramid = 0;
    params.sumTol = 0;
    params.percent_train = 0.7;
    params.numNeighbors = 5;
    params.usekdtree = 0;
end