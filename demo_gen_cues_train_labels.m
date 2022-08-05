%% generate cues with CAM and unsupervised segmentation map

% for the training set 
itype = 'lvwang';
idcls = '0'; % class id
idir = 'train'; % 'test'
unsegpath = ['E:\yinxcao\tuitiantu\datagoogle\', itype, '\unseg\']; % for storing segmentation map
campath = ['E:\yinxcao\tuitiantu\tttcls_google_lvwang\pred\',...
    'regnet040_0.6_gradcam_cues\',idir, '\',itype,'\'];  % for storing CAM 
imgpath = campath; 
respath =campath; % output dir


filelist = dir([imgpath, '*.png']);
num=length(filelist);
parfor i=1:num
    iname = filelist(i).name;
    iname = iname(1:end-4);
    resname = [respath,iname,'_obj.png'];
    if isfile(resname)
        continue;
    end
    unseg = imread([unsegpath, iname,'.png'])+1; % [1,n]
    p = dir([campath, iname,'_',idcls,'_*.tif']);
    campro = imread([campath, p(1).name]); %
    maskall = func_gencue_obj(campro, unseg);
    imwrite(maskall, [respath,iname,'_obj.png']);
    imwrite(maskall*255, [respath,iname,'_objc.png']);
%     subplot(1,2,1);
%     imshow(img);
%     subplot(1,2,2);
%     imshow(maskall*255*0.3+img*0.5,[]);
end



