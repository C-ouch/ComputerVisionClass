% Load image and plot clicked 2D points
figure;                    % create a new figure
imshow(InputImage);         % show the input image
hold on;                    % hold the figure to add more elements
scatter(impoints(:,1), impoints(:,2), 'b', 'filled');  % plot the clicked 2D points in blue

mesh_x = objpoints3D(1,:);   % extract x coordinates of 3D points
mesh_y = objpoints3D(2,:);   % extract y coordinates of 3D points
mesh_z = objpoints3D(3,:);   % extract z coordinates of 3D points

% Apply transformation to 3D points
Xc = R*Xo + t;      % transform the 3D points using rotation matrix R and translation vector t

% Project 3D points onto image
x_estim = K*Xc;     % project the 3D points onto the image plane using camera calibration matrix K
imgpoints2D_estim = [x_estim(1,:)./x_estim(3,:); x_estim(2,:)./x_estim(3,:)]';   % convert the projected points into 2D pixel coordinates

% Plot re-projected 2D points
scatter(imgpoints2D_estim(:,1), imgpoints2D_estim(:,2), 'r', 'filled');   % plot the re-projected 2D points in red

size(impoints)      % display the size of the clicked 2D points
size(imgpoints2D_estim)  % display the size of the re-projected 2D points

% Compute sum-squared distance between actual and estimated 2D points
N = size(impoints, 1);   % get the number of 2D points
error = 0;               % initialize the error to 0
for i = 1:N              % loop over all 2D points
    % compute distance between corresponding points
    d = norm(impoints(i,:) - imgpoints2D_estim(i,:))^2;
    % add distance to total error
    error = error + d;
end

% Print error
fprintf('Sum-squared distance between actual and estimated 2D points: %f\n', error);

% Project transformed mesh onto image
mesh2D = zeros(size(mesh_x));   % initialize an array to store the x-coordinates of the transformed mesh
mesh3D = zeros(size(mesh_y));   % initialize an array to store the y-coordinates of the transformed mesh
for i = 1:numel(mesh_x)         % loop over all points in the mesh
    x = K*(R*[mesh_x(i); mesh_y(i); mesh_z(i)] + t);  % transform the point using R and t and project onto image plane
    mesh2D(i) = x(1)/x(3);     % divide the x-coordinate by the depth to get the pixel x-coordinate
    mesh3D(i) = x(2)/x(3);     % divide the y-coordinate by the depth to get the pixel y-coordinate
end
