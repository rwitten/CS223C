function [] = Classify()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Initialize parameters of SPM classification
params = initParams();

%Build pyramids for each class
train_labels = [];
test_labels = [];
train_filenames = {};
test_filenames = {};
for i=1:params.num_classes
    %Build list of filepaths
    cur_train_image_dir = strcat(params.image_dir, '/train/',params.class_names{i});
    cur_test_image_dir = strcat(params.image_dir, '/test/',params.class_names{i});
    train_fnames = dir(fullfile(cur_train_image_dir, '*.jpg'));
    test_fnames = dir(fullfile(cur_test_image_dir, '*.jpg'));
    num_train_files = size(train_fnames,1);
    num_test_files = size(test_fnames,1);
    newfilenames = cell(num_train_files,1);
    for f = 1:num_train_files
        newfilenames{f} =  strcat('train/',params.class_names{i}, '/', train_fnames(f).name);
    end
    train_filenames(end + (1:num_train_files)) = newfilenames;
    train_labels = [train_labels; i*ones(num_train_files,1)];
    newfilenames = cell(num_test_files,1);
    for f = 1:num_test_files
        newfilenames{f} =  strcat('test/', params.class_names{i}, '/', test_fnames(f).name);
    end
    test_filenames(end + (1:num_test_files)) = newfilenames;
    test_labels = [test_labels; i*ones(num_test_files,1)];
end    
train_filenames = train_filenames';
test_filenames = test_filenames';

% %Separate into training and testing datasets
% train_filenames = {};
% test_filenames = {};
% train_labels = [];
% test_labels = [];
% for i=1:params.num_classes
%     cur_filenames = filenames(totalLabels == i);
%     perm = randperm(size(cur_filenames,1));
%     trainperm = perm(1:floor(params.percent_train*end));
%     testperm = perm(ceil(params.percent_train*end):end);
%     train_filenames = [train_filenames; cur_filenames(trainperm)];
%     test_filenames = [test_filenames; cur_filenames(testperm)];
%     train_labels = [train_labels; i*ones(length(trainperm),1)];
%     test_labels = [test_labels; i*ones(length(testperm),1)];
% end
% clear total_labels;
% clear filenames;
% fprintf('Data separated');


%Note: Codebook technically could be generated from testing data with this
%approach, easy to fix up later though.
train_pyramids = BuildPyramid(train_filenames, params,1);
test_pyramids = BuildPyramid(test_filenames, params,0);

size(train_pyramids)
size(test_pyramids)
fprintf('Pyramids built');
clear filenames;


%Apply kernels if you want
if params.apply_kernel == 1
    train_data = hist_isect_c(train_pyramids, train_pyramids);
    test_data = hist_isect_c(test_pyramids, train_pyramids);
    fprintf('kernel intersect');
elseif params.apply_kernel == 2
    %% perform clustering
    options = foptions;
    options(1) = 1; % display
    options(2) = 1;
    options(3) = 0.01; % precision
    options(5) = 1; % initialization
    options(14) = 100; % maximum iterations
    class_centers = zeros(params.num_classes*params.clustersPerClass, size(train_pyramids,2));
    init_zeros = zeros(params.clustersPerClass, size(train_pyramids,2));
    %% run kmeans for each class
    fprintf('\nRunning k-means\n');
    for i=1:params.num_classes
        class_centers((i-1)*params.clustersPerClass + (1:params.clustersPerClass),:) = ...
            sp_kmeans(init_zeros, train_pyramids(train_labels==i,:), options);
    end
    
    train_data = hist_isect_c(train_pyramids, class_centers);
    test_data = hist_isect_c(test_pyramids, class_centers);
    fprintf('kernel scalable intersect');
else
    train_data = train_pyramids;
    test_data = test_pyramids;
    fprintf('No kernel applied');
end
clear train_pyramids;
clear test_pyramids;




%Train detector
model = train(train_labels,sparse(train_data), '-s 4');
[~, accuracy] = predict(train_labels, sparse(train_data), model);
accuracy

%Test detector
[~, accuracy] = predict(test_labels, sparse(test_data), model);
accuracy %this is correct since we modified dataset to have same size
         %for each class.

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

    params.useNaiveNN = 1;
    
    params.class_names = classes;
    params.num_classes = 15;%length(params.class_names);
    params.clustersPerClass = 5;
    params.max_image_size = 1000;
    params.dictionary_size = 200;
    params.num_texton_images = 150;
    params.pyramid_levels = 4;
    params.max_pooling = 1;
    params.sum_norm = 0;
    params.do_llc = 0;
    params.apply_kernel = 0;
    params.can_skip = 1;
    params.can_skip_sift = 1;
    params.can_skip_calcdict = 1;
    params.can_skip_buildhist = 1;
    params.can_skip_compilepyramid = 1;
    params.sumTol = 0;
    params.percent_train = 0.7;
    params.numNeighbors = 1;
    params.usekdtree = 0;
    params.numPassesSift=10;
end