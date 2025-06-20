clear all

%%%%%%%%%%%%%%%%%%%%%%%%
% Set program parameters
%%%%%%%%%%%%%%%%%%%%%%%%

% name of file
file_name = ['RP23_2022_11_30OD040'];

% number of layers to identify on the image
no_layers = 7;

% % order of polynomial to fit to layers
% max_poly_order = 30;

% set this to 1 if you want to show the smoothed image or to 0 if you 
% want to show the original image 
show_smoothed_image = 0;

% only use a certain subset (1/extract_every) of points in segments 
extract_every = 10;

% read directory
image_dir = '\\winfs-uni.top.gwdg.de\laurence$\My Pictures\RP OCTs\RP_23\OD';
% write directory (for layer points)
write_dir = '\\winfs-uni.top.gwdg.de\laurence$\My Pictures\RP OCTs\RP23 datapoints';
% save directory (for segmented images)
save_dir = '\\winfs-uni.top.gwdg.de\laurence$\My Pictures\RP OCTs\RP23 segmented';
save_file = sprintf('%s\\%s_segmented.tif',save_dir,file_name);
% save directory (for masked images)
mask_dir = '\\winfs-uni.top.gwdg.de\laurence$\My Pictures\RP OCTs\RP23 masked';
save_mask = sprintf('%s\\%s_masked.tif',mask_dir,file_name');
% cropped directory
crop_dir = '\\winfs-uni.top.gwdg.de\laurence$\My Pictures\RP OCTs\RP23 cropped';
crop_file = sprintf('%s\\%s_cropped.tif',crop_dir,file_name');

% filenames for images cropped around segmentation points
pape_crop_file = sprintf('%s\\%s_cropped_pape.tif',crop_dir,file_name');
pape_mask_file = sprintf('%s\\%s_masked_pape.tif',mask_dir,file_name');

% monitor resolution
% set figure size
figsizex = 1200; figsizey = 600;
xpos = 50; ypos = 50;

% set crop limits for image 
xmin = 496; ymax = 496;

% set smoothing parameters - x value should be larger than y
xsmooth = 10; ysmooth = 0.25;

% parameters for edge detection
edge_type = 'canny';
threshold = [0.1,0.2]; smooth_sigma = 3.0;


%%%%%%%%%%%%%%%%
% Start routine
%%%%%%%%%%%%%%%%

% read in image file
image_file = sprintf('%s\\%s.tif',image_dir,file_name);
I = imread(image_file);

% get size of original image
[mm, nn ] = size(I);

% crop down to region of interest
Ic = imcrop(I,[xmin, 1, nn-xmin+1, ymax]);
[mmc,nnc] = size(Ic);

% convert to grayscale
Ic = rgb2gray(Ic);

% smooth image
Ib = imgaussfilt(Ic,[ysmooth,xsmooth]);

% find edge points
E = edge_mod_dola(Ib,edge_type,threshold,smooth_sigma);
[I,J] = find(E);

% plot detected edge points
fplot = figure;
if show_smoothed_image
  imshow(Ib)
else
  imshow(Ic)
end
axis off
hold on
h_ep = plot(J,I,'r.','MarkerSize',2);
hold off
set(gcf,'Position',[xpos,ypos,figsizex,figsizey])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now choose points corresponding to each layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First trace out the segments determined from the edge detection 

% settings for edge following
direction = 'ee';
orient = 'ccw';
nmax = 2000;

for kk = 1:no_layers
  
  h_tit = title(sprintf('Enter start and end points for line segments for layer %d, then press enter',kk)); 
  [xi,yi] = ginputWhite;
  
  % find nearest edge points to inputted points
  [x_out, y_out] = find_nearest_edge_points(E,xi,yi);
  x_start = x_out(1:2:end);
  y_start = y_out(1:2:end);
  x_stop = x_out(2:2:end);
  y_stop = y_out(2:2:end);
  
  % number of segments to trace
  no_segs = floor(length(xi)/2);
  nn = 0;
  xc = zeros(1,nmax); yc = zeros(1,nmax); % preallocate for speed

  % trace out edge segments
  for ii=1:no_segs
    [x_seg,y_seg] = trace_edge_stopstart(E,x_start(ii),y_start(ii),...
        x_stop(ii),y_stop(ii),direction,orient);
      n_points_seg = length(x_seg);
    xc(nn+1:nn+n_points_seg) = x_seg;
    yc(nn+1:nn+n_points_seg) = y_seg;
    nn = nn + n_points_seg;
  end
  % trim unused values
  xc = xc(1:nn); yc = yc(1:nn);
  
  if kk > 1
    delete(hseg)
  end
  hold on
  hseg = plot(xc,yc,'b.','MarkerSize',1);
  hold off
  
  % save edge points
  xe{kk} = xc(1:extract_every:end); ye{kk} = yc(1:extract_every:end);
  
end
delete(hseg)
delete(h_ep)


% Now add additional points for layers where there aren't enough edge
% points

for kk = 1:no_layers
  
  % plot existing points or the layer
%   if kk > 1
%     delete(hseg); delete(hpoint)
%   end
  hold on
  hseg = plot(xe{kk},ye{kk},'yo','MarkerSize',2);
  hold off
  
  h_tit = title(sprintf('Enter additional points for layer %d, then press enter',kk)); 

  % input additional points manually for the layer
  [xi,yi] = ginputWhite;
  hold on
  hpoint = plot(xi,yi,'go','MarkerSize',3);
  hold off
  
  % add points to existing ones
  if ~isempty(xi)
    xe{kk} = [xe{kk}, xi'];
    ye{kk} = [ye{kk}, yi'];
  end
  
end


% show image again with fitted curves
clf
if show_smoothed_image
  imshow(Ib,'Border','tight')
else
  imshow(Ic,'Border','tight')
end
set(gcf,'Position',[xpos,ypos,figsizex,figsizey])

hold on
for kk = 1:no_layers
  
%   if length(xe{kk}) >= max_poly_order
%     poly_order = max_poly_order;
%   else
%     poly_order = length(xe{kk}) - 2;
%   end
  
  % fit spline curve to range of x points
  xx = 1:nnc;
  yy = interp1(xe{kk},ye{kk},xx,'pchip','extrap');
  
%   % fit a polynomial
%   if length(xe{kk}) >= max_poly_order
%     pp = polyfit(xe{kk},ye{kk},max_poly_order);
%     yy = polyval(pp,xx);
%   else
%     yy = spline(xe{kk},ye{kk},xx);
%   end
  
  plot(xx,yy,'-','LineWidth',1)
  
end
hold off
axis off


% create and plot masked image
Imask = create_masked_image(Ic,xe,ye,no_layers);
fmask = figure; imshow(Imask)

% % add lines to masked image
% hold on
% for kk = 1:no_layers
  % fit spline curve to range of x points
  % xx = linspace(min(xe{kk}),max(xe{kk}),500);
  % yy = spline(xe{kk},ye{kk},xx); 
  % plot(xx,yy,'-','LineWidth',1)
% end
% hold off
% axis off



% find limits for cropping images down to segmentation range
for ii = 1:no_layers

    % find limits for cropping
    if ii == 1
        xcmin = min(xe{ii}); xcmax = max(xe{ii});
    else
        if min(xe{ii}) > xcmin
            xcmin = min(xe{ii});
        end
        if max(xe{ii}) < xcmax
            xcmax = max(xe{ii});
        end
    end

end
xcmin = round(xcmin); xcmax = round(xcmax);

% remove parts of the image that would need to extrapolate curves
Imc = Imask(:,xcmin:xcmax);
Icc = Ic(:,xcmin:xcmax);

% save masked image
imwrite(Imc,pape_mask_file,"Compression","none");

% saved cropped image
imwrite(Icc,pape_crop_file,"Compression","none");



% return

% save masked image
%saveas(fmask,save_mask)
imwrite(Imask,save_mask,"Compression","none");

% save image with lines
saveas(fplot,save_file,'tiffn');

% saved cropped image
imwrite(Ic,crop_file,"Compression","none");

% save layer points to file
for kk = 1:no_layers
  
  write_file = sprintf('%s/%s_layer%d.txt',write_dir,file_name,kk);
  A = [xe{kk};ye{kk}];
  
  fid = fopen(write_file,'wt');
  fprintf(fid,'x {pix.]\ty [pix.]\n');
  fprintf(fid,'%f\t%f\n',A);
  fclose(fid);
  
end

