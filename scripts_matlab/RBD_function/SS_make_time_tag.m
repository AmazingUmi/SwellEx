function time_tag = SS_make_time_tag(segment_start_s, segment_end_s, segment_step_s)
%SS_MAKE_TIME_TAG Build a filename-safe tag with integer seconds.
start_s = round(segment_start_s);
end_s = round(segment_end_s);
step_s = round(segment_step_s);
time_tag = sprintf('start_s%d_end_s%d_step_s%d', ...
    start_s, end_s, step_s);
end
