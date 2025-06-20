function [x_out, y_out] = find_nearest_edge_points(E,x_inp,y_inp)
    
% finds nearest edge points in an image to inputted points

% transpose so that first index corresponds to x (horizontal)
E = E';

% get number of points
npoints = length(x_inp);
x_out = zeros(size(x_inp));
y_out = zeros(size(x_inp));

% round to nearest pixel
x_inp = round(x_inp);
y_inp = round(y_inp);

for ii = 1:npoints
  
    % Find edge point closest to inputted point x or y direction
    px = find(E(:,y_inp(ii)));
    py = find(E(x_inp(ii),:));
    [valx,indx] = min(abs(px-x_inp(ii)));
    [valy,indy] = min(abs(py-y_inp(ii)));

    % Make closest point the starting point
    if valx <= valy
        x_nearest = px(indx);
        y_nearest = y_inp(ii);
    else
        x_nearest = x_inp(ii);
        y_nearest = py(indy);
    end

    x_out(ii) = x_nearest;
    y_out(ii) = y_nearest;
    
end