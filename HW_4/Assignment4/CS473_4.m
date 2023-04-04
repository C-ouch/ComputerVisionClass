
M = estimateCameraProjectionMatrix(impoints, objpoints3D);

[K, R, t] = getIntrinsicExtrinsicParams(M);





function M = estimateCameraProjectionMatrix(impoints2D, objpoints3D)
% ESTIMATECAMERAPROJECTIONMATRIX Estimate the camera projection matrix
%   given 2D image points and corresponding 3D object points.
%
%   M = ESTIMATECAMERAPROJECTIONMATRIX(impoints2D, objpoints3D) estimates
%   the camera projection matrix M using a least-squares method based on
%   the given 2D image points (impoints2D) and corresponding 3D object points
%   (objpoints3D).
%
%   Input arguments:
%       - impoints2D: an N-by-2 matrix representing N 2D image points.
%       - objpoints3D: an N-by-3 matrix representing N 3D object points.
%
%   Output argument:
%       - M: a 3-by-4 camera projection matrix that maps 3D object points to
%       2D image points.

    % Build the design matrix P
    N = size(impoints2D, 1);
    P = zeros(N*2, 12);
    for i = 1:N
        X = objpoints3D(i, :);
        x = impoints2D(i, :);
        P(2*i-1:2*i, :) = [
            X(1) X(2) X(3) 1 0 0 0 0 -x(1)*X(1) -x(1)*X(2) -x(1)*X(3) -x(1);
            0 0 0 0 X(1) X(2) X(3) 1 -x(2)*X(1) -x(2)*X(2) -x(2)*X(3) -x(2)
        ];
    end
    
    % Solve for M using a least-squares method
    [U, S, V] = svd(P, "econ");
    q = V(:, end); % This should take the last column V to create q
    M = reshape(q, [4, 3])'; 
    M = M ./ M(3, 4);
end

function [K, R, t] = getIntrinsicExtrinsicParams(M)
% Computes the intrinsic and extrinsic parameters from a camera projection
% matrix M using the method discussed in class.
%
% Inputs:
%   - M: camera projection matrix, in the form [K|R]
%
% Outputs:
%   - K: intrinsic parameters matrix
%   - R: rotation matrix of the object with respect to the camera
%   - t: translation vector of the object with respect to the camera

% Step 1: Find k
k = M(1:3, 1:3) * M(1:3, 1:3)';

% Step 2: Find lambda^2*C
lambda = sqrt(k(3,3));
C = inv(k/lambda^2);

% Step 3: Find R
A = k(1:2, 1:2);
R = zeros(3);
R(1:2, 1:2) = A;
R(3,3) = det(R) / (A(1,1)*A(2,2));

% Step 4: Check sign of R
if R(3,3) < 0
    R(:, 3) = -R(:, 3);
end

% Step 5: Find t
t = inv(k) * M(:, 4);
t = lambda * t(1:3);

% Assign outputs
K = k;
t = t(:); % ensure t is a column vector
end