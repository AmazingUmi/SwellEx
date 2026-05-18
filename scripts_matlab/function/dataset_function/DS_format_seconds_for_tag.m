function tag = DS_format_seconds_for_tag(value_s)
%DS_FORMAT_SECONDS_FOR_TAG Format a seconds value for dataset tags.

if abs(value_s - round(value_s)) < 100 * eps(max(1, abs(value_s)))
    tag = sprintf('%d', round(value_s));
else
    tag = regexprep(sprintf('%.3g', value_s), '\.', 'p');
end
end
