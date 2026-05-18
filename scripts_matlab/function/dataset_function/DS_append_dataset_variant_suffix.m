function dataset_variant_tag = DS_append_dataset_variant_suffix( ...
    dataset_variant_tag, manual_dataset_variant_tag)
%DS_APPEND_DATASET_VARIANT_SUFFIX Append an optional manual suffix to auto tag.

dataset_variant_tag = DS_sanitize_dataset_variant_tag(dataset_variant_tag);
manual_dataset_variant_tag = DS_sanitize_dataset_variant_tag(manual_dataset_variant_tag);

if strlength(manual_dataset_variant_tag) > 0
    dataset_variant_tag = sprintf('%s_%s', ...
        dataset_variant_tag, manual_dataset_variant_tag);
end
end
