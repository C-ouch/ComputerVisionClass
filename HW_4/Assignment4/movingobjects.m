% % We insert a mesh corresponding to the 3d model of the lego object into
% % various locations in an image.
% 
% % We have two images. Each image will have a 3d model inserted using K_lego
% % with three results.
% % We also have three results with K_checker for each of the two images.
% 
% % In order to insert a mesh into the new images (which are found in
% % ./Images/3d_modify, we must transform the 3d points of the mesh Xo to a
% % new mesh called X_transformed using 3D rotations and translations that we
% % choose.
% 
% % % The 3D rotations and translations that we want are as follows.
% % 
% % Rotx = [1 0 0; 0 cosd(45) -sind(45); 0 sind(45) cosd(45)];
% % Translation = [0 20 0]';
% % % In order to get X_transformed, we must first perform our rotations and
% % % translations on Xo and then transform by R and t.
% % 
% % X_transformed = (((Xo * Translation) * Rotx) * R) + t;
% 
% 
% % Define the rotation angle in degrees
% theta_deg = 45;
% 
% % Convert the rotation angle to radians
% theta_rad = deg2rad(theta_deg);
% 
% % Define the rotation matrix around the y-axis
% obj_rot = [cos(theta_rad) 0 sin(theta_rad);
%      0             1 0;
%      -sin(theta_rad) 0 cos(theta_rad)];
% 
% % Define the translation vector
% obj_tran = [0; -3; 2];
% 
% % Apply the rotation and translation
% X_obj_transformed = obj_rot * Xo + obj_tran;
% 
% % Display the original and transformed object coordinates
% figure;
% scatter3(Xo(1,:), Xo(2,:), Xo(3,:), 'b');
% hold on;
% scatter3(X_obj_transformed(1,:), X_obj_transformed(2,:), X_obj_transformed(3,:), 'r');
% axis equal;
% grid on;
% legend('Original object coordinates', 'Transformed object coordinates');
% xlabel('x');
% ylabel('y');
% zlabel('z');
% 
% bkg_im1 = "./Images/3d_modify/generic_1.jpg";
% bkg_im2 = "./Images/3d_modify/generic_2.jpg";
% 
% % Rotate and translate the object in camera coordinates
% X_transformed = (R * X_obj_transformed + t);
% 
% X_tran = X_transformed';
% 
% size(X_transformed)
% 
% % Project the transformed mesh onto the image plane using the camera intrinsic matrix
% X_projected = K * X_tran' ;
% 
% 
% % Project the transformed mesh onto the image plane
% % x_projected = K * (R * X_transformed)' ;
% 
% % Display the image and projected points
% figure;
% imshow(bkg_im1);
% hold on;
% patch('vertices',X_projected(:,1:2), 'facecolor', 'none', 'edgecolor', 'k', 'LineWidth', 2);
% 
% 
% % Plot the original image
% figure;
% imshow(bkg_im1);
% 
% % Calculate the minimum and maximum values of x and y coordinates in X_projected
% min_x = min(X_projected(:,1));
% max_x = max(X_projected(:,1));
% min_y = min(X_projected(:,2));
% max_y = max(X_projected(:,2));
% 
% % Visualize the projected points using scatter
% figure;
% scatter(X_projected(:,1), X_projected(:,2), 30, 'r', 'filled');
% xlim([min_x max_x]);
% ylim([min_y max_y]);
% 
% 

% Define the list of images
image_list = {'./Images/3d_modify/generic_1.jpg', './Images/3d_modify/generic_2.jpg'};

% Define the list of rotation angles in degrees for the three different transforms
rotation_angles = [30, 60, 90];

% Loop through each image
for img_idx = 1:numel(image_list)
    % Read the background image
    bkg_im = imread(image_list{img_idx});

    % Loop through each rotation angle
    for rot_idx = 1:numel(rotation_angles)
        % Define the rotation angle in degrees
        theta_deg = rotation_angles(rot_idx);

        % Convert the rotation angle to radians
        theta_rad = deg2rad(theta_deg);

        % Define the rotation matrix around the y-axis
        obj_rot = [cos(theta_rad) 0 sin(theta_rad);
             0             1 0;
             -sin(theta_rad) 0 cos(theta_rad)];

        % Define the translation vector
        obj_tran = [2; 0; 0];

        % Apply the rotation and translation
        X_obj_transformed = obj_rot * Xo + obj_tran;

        % Rotate and translate the object in camera coordinates
        X_transformed = (R * X_obj_transformed + t);

        % Project the transformed mesh onto the image plane using the camera intrinsic matrix
        X_projected = K * X_transformed;

        % Normalize the projected points
        X_projected(1,:) = X_projected(1,:) ./ X_projected(3,:);
        X_projected(2,:) = X_projected(2,:) ./ X_projected(3,:);

        % Display the image and projected points
        figure;
        imshow(bkg_im);
        hold on;
        patch('Vertices', X_projected(1:2,:)', 'Faces', 1:size(X_projected,2), 'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 2);
        title(sprintf('Image %d - Rotation Angle: %d', img_idx, theta_deg));
    end
end
