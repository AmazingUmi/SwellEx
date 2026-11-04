function options = RBD_validate_multipath_options(options)
%RBD_VALIDATE_MULTIPATH_OPTIONS Validate RBD multipath beam-selection options.

required_fields = ["peak_threshold_db", "min_separation_deg", ...
    "max_num_peaks", "sidelobe_reject_db"];
for field_idx = 1:numel(required_fields)
    field_name = required_fields(field_idx);
    if ~isfield(options, field_name) || isempty(options.(field_name))
        error('rbd_multipath_options.%s must be specified.', field_name);
    end
    field_value = options.(field_name);
    if ~isnumeric(field_value) || ~isscalar(field_value) || ...
            ~isreal(field_value) || ...
            (~isfinite(field_value) && ...
            ~(field_name == "max_num_peaks" && isinf(field_value)))
        error('rbd_multipath_options.%s must be a real scalar.', field_name);
    end
end

if options.min_separation_deg < 0
    error('rbd_multipath_options.min_separation_deg must be nonnegative.');
end
if options.max_num_peaks < 1
    error('rbd_multipath_options.max_num_peaks must be at least 1.');
end
end
