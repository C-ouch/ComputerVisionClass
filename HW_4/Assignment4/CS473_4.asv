load dalekosaur/object.mat

patch('vertices', Xo', 'faces', Faces, 'facecolor', 'w', 'edgecolor', 'k');
axis vis3d;
axis equal;
xlabel('Xo-axis'); ylabel('Yo-axis'); zlabel('Zo-axis');
ObjectDirectory = 'dalekosaur';
InputImage = 'image1.jpg';

[impoints, objpoints3D] = clickPoints( InputImage, ObjectDirectory );

figure;
imshow(InputImage); hold on;
plot( impoints2D(:,1), impoints2D(:,2), 'b.');
%and the object points by calling
figure;
patch('vertices', Xo', 'faces', Faces, 'facecolor', 'w', 'edgecolor', 'k');
axis vis3d;
axis equal;
plot3( objpoints3D(:,1), objpoints3D(:,2), objpoints3D(:,3), 'b.' );

