function [x_new,y_new] = get_new_indices(next_dir,x_old,y_old)

% Returns the x and y indices for the next point in the given direction

if strcmp(next_dir,'nn')
  x_new = x_old; y_new = y_old - 1;
elseif strcmp(next_dir,'ne')
  x_new = x_old + 1; y_new = y_old - 1;
elseif strcmp(next_dir,'ee')
  x_new = x_old + 1; y_new = y_old;
elseif strcmp(next_dir,'se')
  x_new = x_old + 1; y_new = y_old + 1;
elseif strcmp(next_dir,'ss')
  x_new = x_old; y_new = y_old + 1;
elseif strcmp(next_dir,'sw')
  x_new = x_old - 1; y_new = y_old + 1;
elseif strcmp(next_dir,'ww')
  x_new = x_old - 1; y_new = y_old;
elseif strcmp(next_dir,'nw')
  x_new = x_old - 1; y_new = y_old - 1;
else
  error('Unrecognized direction entered: please check')
end

