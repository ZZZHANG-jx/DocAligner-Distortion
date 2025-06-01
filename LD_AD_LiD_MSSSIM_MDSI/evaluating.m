addpath(fullfile(pwd,'SIFTflow/mexDenseSIFT'));
addpath(fullfile(pwd,'SIFTflow/mexDiscreteFlow'));
addpath(fullfile(pwd,'SIFTflow'));
addpath(fullfile(pwd,'dependency'));

folder = '../demo/';
gt_suffix = '_target.png';
uw_suffix = '_dewarpnet.png';
file = fopen('log.txt','w');


paths = dir([folder,'*',uw_suffix]); % xxxx+uw_suffix
mean_ms=0.0;
mean_ld=0.0;
mean_ad=0.0;
mean_lid=0.0;
mean_mdsi=0.0;
for i=1:length(paths)
    % image loading
    uw_name = paths(i).name;
    splitname = strsplit(uw_name,uw_suffix);       % xxxx
    gt_name = [splitname{1},gt_suffix];  % xxxx+gt_suffix
    uw_path = [folder,uw_name]
    gt_path = [folder,gt_name]
    uw = imread(uw_path);
    gt = imread(gt_path);

    % image setting for flow field-based metric
    uw_sift = rgb2gray(uw);
    gt_sift = rgb2gray(gt);
    [rh,rw,~]=size(gt_sift);
    gt_sift=imresize(gt_sift,sqrt(598400/(rh*rw)),'bicubic');
    [rh,rw,rc]=size(gt_sift);
    uw_sift=imresize(uw_sift,[rh rw],'bicubic');

    % msssim, ld, ad, lid
    [ms, ld, ad, lid] = evalUnwarp_msssim_ld_lid_ad(uw_sift,gt_sift);

    % mdsi
    [rows, cols, ~] = size(gt);
    mdsi = MDSI(gt,imresize(uw,[rows,cols]));
    mdsi=0;

    fprintf(file,'%s , MSSSIM: %f, MDSI: %f, LD: %f, AD: %f, Li-D: %f\n',uw_name,ms,mdsi,ld,ad,lid);
    mean_ms=mean_ms+ms;
    mean_ld=mean_ld+ld;
    mean_ad=mean_ad+ad;
    mean_lid=mean_lid+lid;
    mean_mdsi=mean_mdsi+mdsi;
end

mean_ms=mean_ms/length(paths)
mean_ad=mean_ad/length(paths)
mean_ld=mean_ld/length(paths)
mean_lid=mean_lid/length(paths)
mean_mdsi=mean_mdsi/length(paths)

fprintf(file,'---- mean , MSSSIM: %f, MDSI: %f, LD: %f, AD: %f, Li-D: %f----',mean_ms,mean_mdsi,mean_ld,mean_ad,mean_lid);
fprintf('---- mean , MSSSIM: %f, MDSI: %f, LD: %f, AD: %f, Li-D: %f----',mean_ms,mean_mdsi,mean_ld,mean_ad,mean_lid);
fclose(file)
