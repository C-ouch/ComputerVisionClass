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

% Define the list of images
image_list = {'./Images/3d_modify/generic_1.jpg', './Images/3d_modify/generic_2.jpg'};

% Define the list of rotation angles in degrees for the three different transforms
rotation_angles = [20, 65, 105];

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
        obj_roty = [cos(theta_rad) 0 sin(theta_rad);
             0             1 0;
             -sin(theta_rad) 0 cos(theta_rad)];
         
%          x_deg = -15
%          x_rad = deg2rad(x_deg)
         % Define the rotation matrix around the x-axis
        obj_rotx = [1 0 0;
           0 cos(x_rad) -sin(x_rad);
           0 sin(x_rad) cos(x_rad)];

        % Define the translation vector
        obj_tran = [-4; -1; -2];

        % Apply the rotation and translation
        X_obj_transformed = obj_roty * Xo + obj_tran;

        % Rotate and translate the object in camera coordinates
        X_transformed = (R * X_obj_transformed + t);

        % Project the transformed mesh onto the image plane using the camera intrinsic matrix
        X_projected = K_checker2 * X_transformed;

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
