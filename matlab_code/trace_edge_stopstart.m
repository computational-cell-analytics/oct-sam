function [x_seg,y_seg] = ...
    trace_edge_stopstart(I,x_start,y_start,x_stop,y_stop,start_dir,orientation)

% Traces the points along an edge in a binary edge image with specified
% starting point and direction/orientation, until a specified stopping
% point is reached

% 'I' is a binary image (ones for edge pixels, zeros for non-edge pixels)
%
% '(x_start, y_start)' is the location of the edge pixel to start tracing
% from
%
% '(x_stop, y_stop)' is the point at which edge tracing should stop (if too
% close)
%
% 'start_dir' is one of 'nn', 'nw', 'ww', 'sw', 'ss', 'se', 'ee', or 'ne' 
% and is the direction to start looking for the second edge point
%
% 'orientation' is either 'cw' (clockwise) or 'ccw' (counterclockwise) and
% indicates the primary orientation of the geometry

% transpose so that first index corresponds to x (horizontal)
I = I';

% minimum length for the segment before accepting stop point
min_seg_length = 3;

% Initialise segment vectors; pre-allocate to speed things up
seg_length = 1e4; % shouldn't have a segment longer than 10,000 points
x_seg = zeros(1,seg_length);
y_seg = zeros(1,seg_length);
x_seg(1) = x_start;
y_seg(1) = y_start;

% (x_old, y_old) and (x_new, y_new) are the current and next edge point
% indices, respectively
x_old = x_start;
y_old = y_start;

% Order of directions to look for next edge point in
if strcmp(orientation,'cw')
  arrow = {'nn','ne','ee','se','ss','sw','ww','nw'};
else
	arrow = {'nn','nw','ww','sw','ss','se','ee','ne'};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First, check that there is an edge in close to the specified direction

% Set up direction priority based on specified starting direction; be a
% little more stringent than with subsequent searches to avoid user
% specifying wrong direction
start_ind = find(strcmp(arrow,start_dir)); % which index of 'arrow' to start with
poss_dirs_ind = [start_ind, start_ind+1, start_ind+2, start_ind-1];
poss_dirs_ind = mod(poss_dirs_ind-1,8) + 1;
no_dirs = length(poss_dirs_ind);

% Look for next edge point, according to priority just specified
edge_found = 0;
for jj = 1:no_dirs
  [x_new,y_new] = get_new_indices(arrow{poss_dirs_ind(jj)},x_old,y_old);
  if I(x_new,y_new) == 1
    % edge point found in this direction, skip remaining possibilities
    current_dir_ind = poss_dirs_ind(jj);
    edge_found = 1;
    break
  end
end

% Bad choice of start direction if no edge found; return error
if edge_found == 0
	herr = errordlg('No edge found in specified direction')
    waitfor(herr)
    return
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now start from second point and find all remaining points on segment
n = 2; finished = 0;

while finished == 0
    
  % Check whether we're back at the starting point; if so, end loop
  if x_new == x_start && y_new == y_start
      finished = 1;
    break
  end
  
  % Check whether we've reached stopping point; if so, update and end loop.
  % Also enforce a minimum segment size (3 points)
  if (x_new == x_stop && y_new == y_stop) && (n >= min_seg_length)
      x_seg(n) = x_new;
      y_seg(n) = y_new;
      n = n+1;
      finished = 1;
    break
  end
  
  % Update segment vectors
    
  % First, if we have moved along a diagonal check that we haven't 
  % skipped edge pixels (or starting/stopping point) on off-diagonal elements
  dist_sq = (x_new-x_old)^2 + (y_new-y_old)^2;
  if dist_sq == 2 % if diagonal

    % Check that a skipped pixel isn't the starting point; if so, stop
    if (x_new == x_start && y_old == y_start) || (x_old == x_start && y_new == y_start)
      finished = 1;
      break
    end    
    
    % Check that a skipped pixel isn't the stopping point; if so, update
    % and stop
    if (x_new == x_stop && y_old == y_stop) && (n >= min_seg_length)
      x_seg(n) = x_new;
      y_seg(n) = y_old;
      n = n+1;
      finished = 1;
    elseif (x_old == x_stop && y_new == y_stop) && (n >= min_seg_length)
      x_seg(n) = x_old;
      y_seg(n) = y_new;
      n = n+1;
      finished = 1;
    end
      
    % check for edge points on off-diagonal pixels 
    if I(x_new,y_old) == 1
      x_seg(n) = x_new;
      y_seg(n) = y_old;
      x_seg(n+1) = x_new;
      y_seg(n+1) = y_new;
      n = n+2;
    elseif I(x_old,y_new) == 1
      x_seg(n) = x_old;
      y_seg(n) = y_new;
      x_seg(n+1) = x_new;
      y_seg(n+1) = y_new;
      n = n+2;
    else
      x_seg(n) = x_new;
      y_seg(n) = y_new;
      n = n+1;
    end
  else % not a diagonal; simply update segment
    x_seg(n) = x_new;
    y_seg(n) = y_new;
    n = n+1;
  end
    
  % update (x_new, y_new)
  x_old = x_new;
  y_old = y_new;

  % Indices in 'arrow' of possible directions for the next point in 
  %   segment, in order of priority
  poss_dirs_ind = [current_dir_ind, current_dir_ind+1,...
          current_dir_ind+2, current_dir_ind-1, current_dir_ind-2,...
          current_dir_ind+3, current_dir_ind-3];
  poss_dirs_ind = mod(poss_dirs_ind-1,8) + 1;
  no_dirs = length(poss_dirs_ind);

  % Look for next point in segment, according to priority just specified
  finished = 1;
  for jj = 1:no_dirs
      [x_new,y_new] = ...
          get_new_indices(arrow{poss_dirs_ind(jj)},x_old,y_old);
      if I(x_new,y_new) == 1
          % next edge point found, skip remaining possibilities
          current_dir_ind = poss_dirs_ind(jj);
          finished = 0;
          break
      end
  end
    
end

% trim entries without data
x_seg(n:end) = [];
y_seg(n:end) = [];