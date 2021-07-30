# Choroidal-vascularity-index

This is a MATLAB code for batch calculation of the choroidal vascularity index (CVI) in a series of the OCT images.

You need the OCT images marked with yellow arrow in the center of the fovea and marked with red line between the choroid/sclera boudary like below.

![OCT_choroid](https://user-images.githubusercontent.com/56453229/127603648-e2019e0c-c9ed-4c15-b8df-3967ea591ec0.jpg)

This code is based on the 768 x 496 pixels (9200 μm x 1900 μm) sized horizontal OCT images.


It has two parts.

First, the lower border of the hyperreflective band of RPE/Bruch's membrane complex and choroidal area of a predetermined range centered on the fovea are automatically extracted.

![OCT_choroid_cat](https://user-images.githubusercontent.com/56453229/127603887-cb73834d-bae1-4076-8846-67c0de69bd83.jpg)


Second, the image is binarized with Niblack's auto thresholding method, and the CVI is calculated in two ways.

![CVI](https://user-images.githubusercontent.com/56453229/127604213-e0ae9673-44a2-4bf2-8d64-50c47023fad6.jpg)

(Left) original Niblack's thresholding     (Right) despeckling with removal of particles less than 20-pixel size 


Subfoveal choroidal thickness, choroidal area, vessel lumen area, and CVI is calculated and saved in the excel sheet.

The cut and binarized choroidal images are also saved as separate images in the selected folders.


We have adopted the code from Jan Motl for Niblack thresholding. 

(Jan Motl (2021). Niblack local thresholding (https://www.mathworks.com/matlabcentral/fileexchange/40849-niblack-local-thresholding), MATLAB Central File Exchange. Retrieved July 30, 2021.)
