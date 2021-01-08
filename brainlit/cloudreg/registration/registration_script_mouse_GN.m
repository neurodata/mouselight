run('~/CloudReg/registration/map_nonuniform_multiscale_v02_mouse_gauss_newton.m');
atlas_prefix = './atlases/';
atlas_path = [atlas_prefix 'ara_annotation_10um.tif'];
target_path = target_name;
atlas_voxel_size = [10.0, 10.0, 10.0]; % microns
output_path_atlas = [prefix 'labels_to_target_highres.img'];
output_path_target = [prefix 'target_to_labels_highres.img'];
vname = [prefix 'v.mat'];
Aname = [prefix 'A.mat'];
save([prefix 'transform_params.mat'],'atlas_path','atlas_voxel_size','output_path_target','output_path_atlas','nxJ0','dxJ0','dxI','vname','Aname')
transform_data(atlas_path,atlas_voxel_size,Aname,vname,dxI,dxJ0,nxJ0,'target',output_path_atlas,'nearest')
transform_data(target_path,dxJ0,Aname,vname,dxI,atlas_voxel_size,[1320 800 1140],'target',output_path_target,'linear')
