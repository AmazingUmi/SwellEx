function [pair_i, pair_j] = SCM_make_upper_pair_indices(num_elements)
%SCM_MAKE_UPPER_PAIR_INDICES Return upper-triangle pairs including diagonal.
if ~isscalar(num_elements) || num_elements < 1 || num_elements ~= round(num_elements)
    error('num_elements must be a positive integer scalar.');
end

[pair_i, pair_j] = find(triu(true(num_elements), 0));
pair_i = uint16(pair_i(:));
pair_j = uint16(pair_j(:));
end
