%% Data path
data_path = "D:/TEM/3D-GS/r2_gaussian-main/data_generator/real_dataset/FIPS_raw/refcorr/refcorr";
data_save_path = "D:/TEM/3D-GS/r2_gaussian-main/data_generator/real_dataset/FIPS_processed/refcorr";
% data_path = "FIPS_raw/seashell/20211124_seashell_";
% data_save_path = "FIPS_processed/seashell";
% data_path = "FIPS_raw/walnut/20201111_walnut_";
% data_save_path = "FIPS_processed/walnut";


%% Copy text
if ~exist(data_save_path, 'dir')
    mkdir(data_save_path); % Create the folder if it does not exist
end
copyfile(data_path+".txt", data_save_path+"/config.txt");

%% Read data

CtData = create_ct_project(data_path, "3D");
sinogram = CtData.sinogram;

%% Save data
n_imgs = size(sinogram, 2);
for i=1:n_imgs
    img = squeeze(sinogram(:, i, :))';
    img_id = sprintf( '%04d', i) ;
    save(append(data_save_path, "/", img_id, ".mat"), "img");
    fprintf('Saving image %d/%d\n', i, n_imgs);
end

% src = "D:\TEM\3D-GS\r2_gaussian-main\data_generator\real_dataset\FIPS_raw\refcorr";
% dst = "D:\TEM\3D-GS\r2_gaussian-main\data_generator\real_dataset\FIPS_processed\refcorr";
% 
% if ~exist(dst, "dir")
%     mkdir(dst);
% end
% 
% copyfile(fullfile(src, "config.txt"), fullfile(dst, "config.txt"));
% 
% for i = 1:361
%     tif_path = fullfile(src, sprintf("%04d.tif", i));
%     img = single(imread(tif_path));
%     save(fullfile(dst, sprintf("%04d.mat", i)), "img");
%     fprintf("Saved %04d.mat\n", i);
% end