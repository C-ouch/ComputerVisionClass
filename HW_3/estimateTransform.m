function H = estimateTransform( pts1, pts2 )

% D = design matrix of size P x 9

if size(D,1) >= 9
    [U,S,V] = svd(D,'econ');
else % D = 8x9 -- RANSAC estimateTransform
    [U,S,V] = svd(D);
end

q = V(:,1);

% reshape q PROPERLY to get H

end