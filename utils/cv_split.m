function [train_data, val_data] = cv_split(x, y, val_pct)
% Split data into training and validation sets.

n_pts = size(x, 1);
n_val_pts = ceil(n_pts * val_pct);
val_idx = false(n_pts, 1);
val_idx(randsample(n_pts, n_val_pts)) = true;

val_data.x = x(val_idx, :);
val_data.y = y(val_idx, :);

train_data.x = x(~val_idx, :);
train_data.y = y(~val_idx, :);

end
