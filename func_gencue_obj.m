function camobj_mask = func_gencue_obj(campro, unseg)
% campro: cam
% unseg: unsupervised segmentation map
bd = boundarymask(unseg);% imshow(1-bd,[]);
cc = bwconncomp(1-bd); % remove boundary
L = labelmatrix(cc); % imagesc(L);
numclass = length(unique(L(:)));
camobj = zeros(size(campro),'single');
for i=1:numclass
    id = find(unseg(:)==i);
    camobj(id) = mean(campro(id)); % mean value of each object
end
% imagesc(camobj);
t = graythresh(camobj); % otsu thresholding
camobj_mask = uint8(camobj>t); 
% imshow(camobj_mask*255*0.3+img*0.5,[]);
end