function dataset_variant_tag = DS_sanitize_dataset_variant_tag(dataset_variant_tag)
%DS_SANITIZE_DATASET_VARIANT_TAG Make a user suffix safe for paths.
dataset_variant_tag = string(dataset_variant_tag);
dataset_variant_tag = strtrim(dataset_variant_tag);

if strlength(dataset_variant_tag) == 0
    return;
end

dataset_variant_tag = regexprep(dataset_variant_tag, '[^A-Za-z0-9_-]+', '_');
dataset_variant_tag = regexprep(dataset_variant_tag, '_+', '_');
dataset_variant_tag = regexprep(dataset_variant_tag, '^_|_$', '');
end
