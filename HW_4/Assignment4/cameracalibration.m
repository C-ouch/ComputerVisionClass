% This script stores information obtained from the Camera Calibration tool
% in MATLAB. The K_checker value stores camera intrinsics from a
% checkerboard pattern that is on a wall surface. K_checker2 is another
% attempt using the same method.

% K_checker = [
% fx s cx;
% 0 fy cy;
% 0 0 1];

K_checker = [
3621.58927379455	0	2044.51754698402;
0	3780.03555950298	1101.00246641569;
0	0	1];

K_checker2 = [
3233.90486110448	0	3011.60015182354;
0	3239.46505111660	1595.07276889265;
0	0	1];

% We can compare to the K value of the lego object:
K_lego = [
12552343.2291599	5174512.31781082	1894.38941669120;
5174512.31781082	10524037.7552560	1412.07507622200;
1894.38941669120	1412.07507622200	1];

%The reason why the K values obtained from the LEGO object and the checkerboard
% are different is that the two calibration methods use different types
% of calibration targets, and the characteristics of these targets affect 
% the calibration results.

% The LEGO object is a 3D object with complex shapes, textures, and structures, 
% making it difficult to extract accurate and reliable calibration information. 
% Moreover, the camera calibration algorithm used in the LEGO experiment may not
% have been able to handle the complexities of the 3D object calibration target,
% resulting in the larger values in the K matrix obtained.

% On the other hand, the checkerboard pattern is a simple planar object
% with a regular grid of corners. It is easier to extract accurate and
% reliable calibration information from this type of calibration target.
% The camera calibration algorithm used in the checkerboard experiment was
% likely able to handle the simpler target better, resulting in a more accurate
% and reliable calibration.

% For K_checker:
% 
% fc = [3621.58927379455, 3780.03555950298]
% cc = [2044.51754698402, 1101.00246641569]
% alpha_c = 0 (since the skew coefficient K(1,2) is zero)
% For K_checker2:
% 
% fc = [3233.90486110448, 3239.46505111660]
% cc = [3011.60015182354, 1595.07276889265]
% alpha_c = 0 (since the skew coefficient K(1,2) is zero)
% For K_lego:
% 
% fc = [12552343.2291599, 10524037.7552560]
% cc = [1894.38941669120, 1412.07507622200]
% alpha_c = 1 (since the skew coefficient K(1,2) is not zero)