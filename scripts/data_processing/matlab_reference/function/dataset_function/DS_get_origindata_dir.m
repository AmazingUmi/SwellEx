function origindata_dir = DS_get_origindata_dir(project_dir)
%DS_GET_ORIGINDATA_DIR Return the root directory for source/original data.
origindata_dir = fullfile(project_dir, 'Origindata');

if ~isfolder(origindata_dir)
    error('Origindata directory not found: %s', origindata_dir);
end
end
