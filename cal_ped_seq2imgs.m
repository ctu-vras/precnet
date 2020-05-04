%Transform seq files of Caltech Pedestrian Dataset - test part (Dollar et al. 2009, http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) into images.
%Download and install Piotr's Computer Vision Matlab Toolbox (https://pdollar.github.io/toolbox/) first.
names_set={'set06','set07','set08','set09','set10'};
sets_last_idx=[18,11,10,11,11];

root_file='./datasets/raw_cal_ped_dataset/';
addpath(root_file)

for set_idx=1:numel(names_set)
    name_set=names_set{set_idx};
    size=sets_last_idx(set_idx);

for num_seq=0:size
name_seq=['V',sprintf('%03d',num_seq)];

source=[root_file, name_set];
mkdir([root_file, 'imgs/', name_set], name_seq)
save_dest=[root_file,'/imgs/',name_set,'/',name_seq];
Is = seqIo([source,'/',name_seq,'.seq'], 'toImgs', [save_dest], [], [], []);
end

end
