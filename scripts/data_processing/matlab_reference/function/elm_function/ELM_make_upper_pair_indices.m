function [pair_i, pair_j] = ELM_make_upper_pair_indices(num_elements)
%ELM_MAKE_UPPER_PAIR_INDICES Return strict upper-triangle element pairs.

if ~isscalar(num_elements) || ~isnumeric(num_elements) || ...
        ~isfinite(num_elements) || num_elements < 2 || ...
        num_elements ~= round(num_elements)
    error('num_elements must be an integer scalar greater than or equal to 2.');
end

[pair_i, pair_j] = find(triu(true(num_elements), 1));
pair_i = uint16(pair_i(:));
pair_j = uint16(pair_j(:));
end
