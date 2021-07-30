clear all;
close all; 
clc; 

%% File list exploration

Filename = dir ('*.tif');
Filename = struct2cell(Filename);

S = size(Filename);
S = S(2);

for i=1:S
    List {1,i} = Filename{1,i};
end

for A=1:S


    %% #1 extraction

    a = imread(List{A});                        %% yellow arrow pointing foveal center / choroid boundary in red
    
    intensity = mean(a(:))/55;   %% intensity = mean(a(:))/55;
    if intensity >0.75
        intensity=0.75;
    end
    
    red = im2double(a(:,:,1));
    green = im2double(a(:,:,2));
    blue = im2double(a(:,:,3));

    label = green - blue;
    label (label<0.5)=0;    %% 1500 um labeling // label<0.5
    
    range = 3000;           %% range in um, you can choose any range you want to analyze
    rangepix = ceil(range/2*768/9200);   %% half range in pixel / 768 pix = 9200 um
    
    [YL, XL] = find(label);
    centerX = ceil((min(XL) + max(XL))/2);
    rangeX = [centerX-rangepix:centerX+rangepix];
    notrangeX = setdiff([1:768], rangeX);
    
    choroid = red - green;
    choroid (choroid<0.9)=0;  %% choroid boundary extraction  // choroid<0.9
    choroid = im2bw(choroid);
    
    [YC, XC] = find(choroid);    

   
    
    %% RPE extraction method #1
    
    blue (blue<intensity)=0;     %% RPE extraction
    RPE = im2bw(blue);
    RPE2 = bwareaopen(RPE, 500, 4);  %% RPE despeckling  // bwareaopen(RPE, 500, 4)

    for i=435:490           %% remove the L shaped scale bar in the left lower corner
        for j= 10:50
            RPE2(i,j)=0;
        end
    end

    for i=1:768                       %% RPE lower border detection
        k = RPE2(496*i-495:496*i)';
        k = double(k);
        if max(k)==0
            RPE3(:,i)=k;
        else
            index = find(k);
            maxindex = max(index);
            k(maxindex)=2;
            k(k<2)=0;
            k(k==2)=1;
            RPE3(:,i)=k;
        end
    end

    RPE4 = bwareaopen(RPE3, 25, 8);  %% RPE despeckling // bwareaopen(RPE3, 25, 8) 

    nhood = ones(10,10);               %% connecting a broken line by dilation and skeletonization // nhood = ones(10,10)
    RPE5 = imdilate(RPE4, nhood);
    RPE6 = bwskel(RPE5);


    %% Add line

    [Y,X] = find(bwmorph(RPE6,'endpoints'));

    S = size(Y);
    S = S(1);

    Add = zeros(496, 768);

    for i=1:2:S-3
    X1 = X(i+1);
    X2 = X(i+2);
    Y1 = Y(i+1);
    Y2 = Y(i+2);
    slope = (Y1-Y2)/(X2-X1);
    X_Add=[];
    Y_Add=[];
        for j=1:X2-X1
            X_Add(j) = X1+j;
            Y_Add(j) = Y1-floor(slope*j);
            Add(Y_Add(j), X_Add(j))=1;
        end   
    end

    RPE7 = RPE6 + Add;

    Extract1 = RPE7;
    Extract1 = imbinarize(Extract1);


    %% #2 extraction

    blue = im2double(a(:,:,3));
    RRPE=zeros(496, 768);

    for i=1:1:768
        L = blue(496*i-495:496*i);
        Lmax = min(maxk(L,6));
        L(L<Lmax)=0;
        L(L>=Lmax)=1;
        L=L';
        RRPE(:,i)=L;       
    end

    for i=430:496           %% remove the L shaped scale bar in the left lower corner
        for j= 1:50
            RRPE(i,j)=0;
        end
    end

    for i=1:768                       %% RPE lower border detection
        k = RRPE(496*i-495:496*i)';
        k = double(k);
        if max(k)==0
            RRPE2(:,i)=k;
        else
            index = find(k);
            maxindex = max(index);
            k(maxindex)=2;
            k(k<2)=0;
            k(k==2)=1;
            RRPE2(:,i)=k;
        end
    end

    %% pixel location average calculation

    [Y, X] = find(RRPE2);
    S2 = size(Y);
    S2 = S2(1);

    for i=1:14
        avg(i) = mean(Y(1:29));
        med(i) = median(Y(1:29));
        st(i) = std(Y(1:29));
    end

    for i=S2-13:S2
        avg(i) = mean(Y(S2-28:S2));
        med(i) = median(Y(S2-28:S2));
        st(i) = std(Y(S2-28:S2));
    end


    for i=15:S2-14
        avg(i) = mean(Y(i-14:i+14));
        med(i) = median(Y(i-14:i+14));
        st(i) = std(Y(i-14:i+14));
    end


    %% pixel selection by Avg and Stdev

    for i=1:S2                              %% upper 1.5 / lower 0.7
        upper(i) = avg(i) + 1.5*st(i);
        lower(i) = avg(i) - 0.7*st(i);
    end

    %% Outlier selection

    Q = zeros(1,S2);

    for i=1:S2
        if Y(i) >= upper(i)
            Q(i) = X(i);
        elseif Y(i) <= lower(i)
            Q(i) = X(i);
        end
    end


    %% Outlier removal

    S3 = size(Q);
    S3 = S3(2);

    RRPE3 = RRPE2;

    for i=1:S3
        if Q(i)>0
        RRPE3(:,Q(i))=0;
        end        
    end

    %% image dilation to fill the gap
    se1=strel('line',3,0);
    se2=strel('line',2,90);

    RRPE4=imdilate(RRPE3,se1);
    RRPE4=imdilate(RRPE4,se2);

    for i=1:6                   %%for i=1:6
    RRPE4=imdilate(RRPE4,se1);
    RRPE4=imdilate(RRPE4,se2);
    end

    RRPE4=imbinarize(RRPE4);
    RRPE4=bwskel(RRPE4);

    RRPE4 = bwareaopen(RRPE4, 50, 8);

    Extract2 = RRPE4;

   

    %% Merge

    for i=1:768                      
        k1 = Extract1(496*i-495:496*i)';
        k2 = Extract2(496*i-495:496*i)';
        k1 = double(k1);
        k2 = double(k2);

        if and(max(k1)==0, max(k2)>0)
            fRPE(:,i)=k2;
        else
            fRPE(:,i)=k1;
        end
    end

    fRPE = bwareaopen(fRPE, 200, 8);   %% fRPE = bwareaopen(fRPE, 200, 8);
    
    [YR, XR] = find(fRPE);
    
   %% image masking
    
    % 1500 um range
    mask1 = im2gray(a);
    mask1(:,notrangeX) = 0;
    
    % choroid masking
    mask2 = mask1;
  
    S = size(XC);
    S = S(1);
    for i=1:S
        mask2(YC(i):end,XC(i))=0;
    end
    
    % RPE masking
    mask3 = mask2;
    
    S = size(XR);
    S = S(1);
    for i=1:S
        mask3(1:YR(i),XR(i))=0;
    end

    % Choriocapillary masking
    mask4 = mask3;
    RPE_dist = 200;    %% distance from RPE line in um
    RPE_dist_pix = ceil(RPE_dist*768/9200);
    
    for i=1:S
        mask4(YR(i)+RPE_dist_pix:end,XR(i))=0;
    end
    
    
    % choroid area boundary
    [YM, XM] = find(mask3);
    
    
    
    %% Analysis
    
    SCT(A,1) = nnz(mask3(:,centerX))*1900/496;    %% 496 pix = 1900 um
    SCTA(A,1) = nnz(mask3)*1900*9200/496/768;     %% horizontal 768 pix = 9200 um / vertical 496 pix = 1900 um
     
    
    %% Niblack
    
    aa = niblack(im2gray(a), [25 25], 0.1);       % niblack, [25,25] range, pixel >  mean + k * standard_deviation - c => 'k=0.2', 'c=0 (default)' // ([20 20], 0.2)

    mask1_2 = im2gray(aa);
    mask1_2(:,notrangeX) = 0;
    
    % choroid masking
    mask2_2 = mask1_2;
  
    S = size(XC);
    S = S(1);
    for i=1:S
        mask2_2(YC(i):end,XC(i))=0;
    end
    
    % RPE masking
    mask3_2 = mask2_2;
    
    S = size(XR);
    S = S(1);
    for i=1:S
        mask3_2(1:YR(i),XR(i))=0;
    end
    
    bb = bwareaopen(im2bw(mask3_2), 20, 4);          % niblack -> despeckle  // (20, 4)
    
%     montage({aa, mask3, mask3_2, bb})
    
    choroid_range = mask3;
    choroid_range(choroid_range>0)=255;
    choroid_range = bwmorph(choroid_range, 'close');
    
    CA(A,1) = nnz(choroid_range);
    VA1(A,1) = CA(A,1)-nnz(mask3_2);
    VA2(A,1) = CA(A,1)-nnz(bb);
    CVI1(A,1) = VA1(A,1)/CA(A,1);
    CVI2(A,1) = VA2(A,1)/CA(A,1);
    
    %% Image display

    fRPE2 = im2uint8(fRPE);

    red3 = a(:,:,1);                %% image concatenation
    green3 = a(:,:,2);
    blue3 = a(:,:,3);

    red3(fRPE2==255)=0;
    green3(fRPE2==255)=0;
    blue3(fRPE2==255)=0;

    choroid_border = edge(choroid_range);
    choroid_border = im2uint8(choroid_border);
     
    red3(choroid_border==255)=0;
    green3(choroid_border==255)=0;
    blue3(choroid_border==255)=0;
    
    red4 = red3+fRPE2;
    green4 = green3+choroid_border;
    blue4 = blue3+fRPE2+choroid_border;

    concat = cat(3, red4, green4, blue3); 
        

    %% Image save
    filename = strcat(List{A}(1:end-4),'_cat.tif');
    filename2 = strcat(List{A}(1:end-4),'_cut.tif');
    filename3 = strcat(List{A}(1:end-4),'_CVI1.tif');
    filename4 = strcat(List{A}(1:end-4),'_CVI2.tif');
    
    F = fullfile('C:\', 'test', '1_choroid', filename);         %% you can change the directory for file saving
    F2 = fullfile('C:\', 'test', '2_choroid cut', filename2);
    F3 = fullfile('C:\', 'test', '3_CVI1', filename3);
    F4 = fullfile('C:\', 'test', '4_CVI2', filename4);
    
    imwrite(concat,F);
    imwrite(mask3,F2);
    imwrite(mask3_2,F3);
    imwrite(bb,F4);

end


%% Excel export

warning( 'off', 'MATLAB:xlswrite:AddSheet' ) ;

header1={'Image #','choroid area','vessel area1','vessel area2','CVI1', 'CVI2','SCT','SCTA'}; 

writecell(header1,'Analysis.xlsx','Range','A1');


writecell(List','Analysis.xlsx','Range','A2');     
writematrix(CA,'Analysis.xlsx','Range','B2');    
writematrix(VA1,'Analysis.xlsx','Range','C2'); 
writematrix(VA2,'Analysis.xlsx','Range','D2'); 
writematrix(CVI1,'Analysis.xlsx','Range','E2'); 
writematrix(CVI2,'Analysis.xlsx','Range','F2'); 
writematrix(SCT,'Analysis.xlsx','Range','G2'); 
writematrix(SCTA,'Analysis.xlsx','Range','H2'); 

%% NIBLACK local thresholding functions adopted from Jan Motl (2021)
% Jan Motl (2021). Niblack local thresholding (https://www.mathworks.com/matlabcentral/fileexchange/40849-niblack-local-thresholding), MATLAB Central File Exchange. Retrieved July 30, 2021.

function output = niblack(image, varargin)
% Initialization
numvarargs = length(varargin);      % only want 4 optional inputs at most
if numvarargs > 4
    error('myfuns:somefun2Alt:TooManyInputs', ...
     'Possible parameters are: (image, [m n], k, offset, padding)');
end
 
optargs = {[3 3] -0.2 0 'replicate'};   % set defaults
 
optargs(1:numvarargs) = varargin;   % use memorable variable names
[window, k, offset, padding] = optargs{:};

if ndims(image) ~= 2
    error('The input image must be a two-dimensional array.');
end

% Convert to double
image = double(image);

% Mean value
mean = averagefilter(image, window, padding);

% Standard deviation
meanSquare = averagefilter(image.^2, window, padding);
deviation = (meanSquare - mean.^2).^0.5;

% Initialize the output
output = zeros(size(image));

% Niblack
output(image > mean + k * deviation - offset) = 1;
end

function image=averagefilter(image, varargin)
%AVERAGEFILTER 2-D mean filtering.
%   B = AVERAGEFILTER(A) performs mean filtering of two dimensional 
%   matrix A with integral image method. Each output pixel contains 
%   the mean value of the 3-by-3 neighborhood around the corresponding
%   pixel in the input image. 
%
%   B = AVERAGEFILTER(A, [M N]) filters matrix A with M-by-N neighborhood.
%   M defines vertical window size and N defines horizontal window size. 
%   
%   B = AVERAGEFILTER(A, [M N], PADDING) filters matrix A with the 
%   predefinned padding. By default the matrix is padded with zeros to 
%   be compatible with IMFILTER. But then the borders may appear distorted.
%   To deal with border distortion the PADDING parameter can be either
%   set to a scalar or a string: 
%       'circular'    Pads with circular repetition of elements.
%       'replicate'   Repeats border elements of matrix A.
%       'symmetric'   Pads array with mirror reflections of itself. 
%
%   Comparison
%   ----------
%   There are different ways how to perform mean filtering in MATLAB. 
%   An effective way for small neighborhoods is to use IMFILTER:
%
%       I = imread('eight.tif');
%       meanFilter = fspecial('average', [3 3]);
%       J = imfilter(I, meanFilter);
%       figure, imshow(I), figure, imshow(J)
%
%   However, IMFILTER slows down with the increasing size of the 
%   neighborhood while AVERAGEFILTER processing time remains constant.
%   And once one of the neighborhood dimensions is over 21 pixels,
%   AVERAGEFILTER is faster. Anyway, both IMFILTER and AVERAGEFILTER give
%   the same results.
%
%   Remarks
%   -------
%   The output matrix type is the same as of the input matrix A.
%   If either dimesion of the neighborhood is even, the dimension is 
%   rounded down to the closest odd value. 
%
%   Example
%   -------
%       I = imread('eight.tif');
%       J = averagefilter(I, [3 3]);
%       figure, imshow(I), figure, imshow(J)
%
%   See also IMFILTER, FSPECIAL, PADARRAY.

%   Contributed by Jan Motl (jan@motl.us)
%   $Revision: 1.2 $  $Date: 2013/02/13 16:58:01 $


% Parameter checking.
numvarargs = length(varargin);
if numvarargs > 2
    error('myfuns:somefun2Alt:TooManyInputs', ...
        'requires at most 2 optional inputs');
end
 
optargs = {[3 3] 0};            % set defaults for optional inputs
optargs(1:numvarargs) = varargin;
[window, padding] = optargs{:}; % use memorable variable names
m = window(1);
n = window(2);

if ~mod(m,2) m = m-1; end       % check for even window sizes
if ~mod(n,2) n = n-1; end

if (ndims(image)~=2)            % check for color pictures
    display('The input image must be a two dimensional array.')
    display('Consider using rgb2gray or similar function.')
    return
end

% Initialization.
[rows columns] = size(image);   % size of the image

% Pad the image.
imageP  = padarray(image, [(m+1)/2 (n+1)/2], padding, 'pre');
imagePP = padarray(imageP, [(m-1)/2 (n-1)/2], padding, 'post');

% Always use double because uint8 would be too small.
imageD = double(imagePP);

% Matrix 't' is the sum of numbers on the left and above the current cell.
t = cumsum(cumsum(imageD),2);

% Calculate the mean values from the look up table 't'.
imageI = t(1+m:rows+m, 1+n:columns+n) + t(1:rows, 1:columns)...
    - t(1+m:rows+m, 1:columns) - t(1:rows, 1+n:columns+n);

% Now each pixel contains sum of the window. But we want the average value.
imageI = imageI/(m*n);

% Return matrix in the original type class.
image = cast(imageI, class(image));
end


