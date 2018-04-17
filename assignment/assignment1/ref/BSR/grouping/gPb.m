function edge = gPb(imgFile,k)

%% Compute globalPb and hierarchical segmentation for an example image.

addpath(fullfile(pwd,'lib'));

%% 1. compute globalPb on a BSDS image (5Gb of RAM required)

[gPb_orient, gPb_thin, ~] = globalPb(imgFile, '');

if k == 0
    edge = gPb_thin;
    edge = uint8((edge>0.3).*255);
else
    %% 2. compute Hierarchical Regions

    % for boundaries
    ucm = contours2ucm(gPb_orient, 'imageSize');

    %% 3. some threshold
    bdry = (ucm >= k);

    edge = uint8(bdry.*255);
end