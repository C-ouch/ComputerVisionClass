% Load image and plot clicked 2D points
figure;
imshow(InputImage);
hold on;
scatter(impoints(:,1), impoints(:,2), 'b', 'filled');

mesh_x = objpoints3D(1,:);
mesh_y = objpoints3D(2,:);
mesh_z = objpoints3D(3,:);

% Define 3D points of the mesh
X1 = [mesh_x(:)'; mesh_y(:)'; mesh_z(:)'];

% Apply transformation to 3D points
Xc = R*X1 + t;

% Project 3D points onto image
x_estim = K*Xc;
imgpoints2D_estim = [x_estim(1,:)./x_estim(3,:); x_estim(2,:)./x_estim(3,:)]';

% Plot re-projected 2D points
scatter(imgpoints2D_estim(:,1), imgpoints2D_estim(:,2), 'r', 'filled');


size(impoints)
size(imgpoints2D_estim)

% Compute sum-squared distance between actual and estimated 2D points
error = sum(sum((impoints - imgpoints2D_estim).^2));

% Print error
fprintf('Sum-squared distance between actual and estimated 2D points: %f\n', error);

% % Compute sum-squared distance between actual and estimated 2D points
% error = 0;
% for i = 1:size(impoints,1)
%     error = error + sum((impoints(i,:) - imgpoints2D_estim(i,:)).^2);
% end
% 
% % Print error
% fprintf('Sum-squared distance between actual and estimated 2D points: %f\n', error);

% Project transformed mesh onto image
mesh2D = zeros(size(mesh_x));
for i = 1:numel(mesh_x)
    x = K*(R*[mesh_x(i); mesh_y(i); mesh_z(i)] + t);
    mesh2D(i) = x(1)/x(3);
    mesh3D(i) = x(2)/x(3);
end

% Plot transformed mesh on image
figure;
imshow(I); hold on; % ‘hold on’ holds the image to draw more content
patch('vertices', x', 'faces', Faces, 'facecolor', 'n', 'edgecolor', 'b');



figure;
imshow(InputImage);
hold on;
scatter(mesh2D, mesh3D, 5, 'r', 'filled');