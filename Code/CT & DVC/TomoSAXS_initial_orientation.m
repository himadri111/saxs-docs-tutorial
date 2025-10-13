%% TomoSAXS fibre orientation from CT Data
% Alissa Parmenter 06/10/2024
% Input: fibre tracing xml from Avizo 
% Outputs:  .csv file containg x,y,z,theta,phi for points along the fibre
%           .txt pointcloud for DVC
%           fibre space curve data matrix

tic

input_filename = '167208_fibres.xml'; % xml file exported from Avizo fibre tracing module
pointcloud_output_filename = '167208_dvc_pointcloud.txt'; % name of pointcloud for DVC analysis
orientation_output_filename = '167208_orientation.csv'; % name of output csv file containing orientation information for TomoSAXS reconstruction
pointcloud_space_curve_filename = '167208_pointcloud_spacecurve'; % name of spacecurve .mat for input into DVC metrics

%% Fibre tracing to pointcloud
% Place evenly spaced points along fibres

PointID_range = 'P:P'; % Avizo 2022: R:R; Avizo 2023: P:P
coord_range = 'C:E';
lamella_range = 'S:S';
inc = 4;

%% Import coord PointIDs L
[coord, ~, ~] = xlsread(input_filename, 'Points', coord_range);
[~, PointIDs, ~] = xlsread(input_filename, 'Segments', PointID_range);
[L, ~, ~] = xlsread(input_filename, 'Segments', lamella_range);
PointIDs(1, :) = []; % Remove header

N = size(PointIDs, 1); % Number of fibers

% Preallocate estimated space for point_cloud
point_cloud = zeros(N * 1000, 10);  % Estimate large enough initial space
pc_count = 0;  % Keep track of actual number of points

for i = 1:N
    a = str2num(PointIDs{i, 1})';  % Convert PointIDs to array
    n = length(a);                 % Number of points in fibre

    % Preallocate fibre coordinates for current fibre
    fibre = coord(a + 1, :);        % Extract coordinates for fibre

    %% Interpolate points along fibre
    pos = fibre(1, :);              % Start with the first point
    p1 = fibre(2, :);               % Second point
    dir = (p1 - pos) / norm(p1 - pos);  % Initial direction vector

    for p = 2:n-1
        p0 = pos(end, :);
        dist = norm(p1 - p0);       % Distance between latest stored and current point

        while inc < dist
            % Calculate direction and new point
            dir_new = (p1 - p0) / norm(p1 - p0);
            p_new = p0 + inc * dir_new;  % Create new point at distance 'inc'
            pos = [pos; p_new];          % Store new point
            dir = [dir; dir_new];        % Store direction vector

            dist = norm(p1 - p_new);     % Update remaining distance
            p0 = p_new;                  % Update last point
        end

        p1 = fibre(p + 1, :);            % Move to the next point
    end

    %% Store fibre's interpolated points in the point cloud
    l = 1; % Lamella number (can be replaced with actual if needed)
    num_pos = size(pos, 1);  % Number of interpolated points

    fpc = [(ones(num_pos, 1) * l), ...
           (ones(num_pos, 1) * i), ...
           (1:num_pos)', ...
           (pc_count + 1:pc_count + num_pos)', ...
           pos, dir];  % Create row for point cloud
    
    % Add the points for this fibre to the point_cloud matrix
    point_cloud(pc_count + 1:pc_count + num_pos, :) = fpc;
    pc_count = pc_count + num_pos;  % Update total point count

    % Reset variables for the next iteration
    pos = [];
    dir = [];
end

% Trim point_cloud to actual size
point_cloud = point_cloud(1:pc_count, :);

% Write to file
writematrix(point_cloud(:, 4:7), pointcloud_output_filename, 'Delimiter', 'tab');

%% space curve fitting - fit 3rd order polynomial to data
% Number of fibers
fibers_idx = unique(point_cloud(:, 2));  % Extract unique fiber indices
nfibres = length(fibers_idx);            % Number of fibers

% Preallocate cell arrays and variables
fx = cell(nfibres, 1);
fy = cell(nfibres, 1);
fz = cell(nfibres, 1);
s_linear = [];

parfor i = 1:nfibres
    % Extract data for the current fiber
    fiber_id = fibers_idx(i);
    fiber_idx = (point_cloud(:, 2) == fiber_id);
    fiber_data = point_cloud(fiber_idx, :);
    npts = size(fiber_data, 1); % Number of points in the fiber

    if npts < 4
        % Assign empty values for fibers with fewer than 4 points
        fx{i} = [];
        fy{i} = [];
        fz{i} = [];
        s_linear = [s_linear; NaN(npts, 1)];
    else
        % Calculate distances for s_0
        p0 = fiber_data(1:end-1, 5:7);
        p1 = fiber_data(2:end, 5:7);
        distances = sqrt(sum((p1 - p0).^2, 2));  % Vectorized distance calculation
        s_0 = [0; cumsum(distances)];            % Cumulative distance for parameter s

        % Fit the point cloud data using poly3 (degree 3 polynomial)
        fx_0 = polyfit(s_0, fiber_data(:, 5), 3);  % Fit x-coordinates
        fy_0 = polyfit(s_0, fiber_data(:, 6), 3);  % Fit y-coordinates
        fz_0 = polyfit(s_0, fiber_data(:, 7), 3);  % Fit z-coordinates

        % Store results
        fx{i} = fx_0;
        fy{i} = fy_0;
        fz{i} = fz_0;

        % Append s_0 to s_linear
        s_linear = [s_linear; s_0];
    end
end

%% tangent calculations
t = [];  % Preallocate tangent matrix
for i = 1:nfibres
    npts = sum(point_cloud(:, 2) == fibers_idx(i)); % Number of points in fibre

    if npts < 4
       t = [t; NaN(npts, 3)];
    else
        % Retrieve coefficients for polynomials
        coeff_fx = fx{i};
        coeff_fy = fy{i};
        coeff_fz = fz{i};
        s_0 = s_linear(point_cloud(:, 2) == fibers_idx(i), 1);

        % Tangent calculation
        t0 = [3*coeff_fx(1)*s_0.^2 + 2*coeff_fx(2)*s_0 + coeff_fx(3), ...
              3*coeff_fy(1)*s_0.^2 + 2*coeff_fy(2)*s_0 + coeff_fy(3), ...
              3*coeff_fz(1)*s_0.^2 + 2*coeff_fz(2)*s_0 + coeff_fz(3)];
        t = [t; t0];
    end
end

%% f_orientation
% Define reference direction for orientation calculation
ref_dir1 = t;
ref_dir1(:, 3) = 0; % z=0 to create a plane

if sum(sign(t(:, 1)) == sign(t(:, 2))) / length(t) > 0.5
    ref_dir1(:, 1) = abs(ref_dir1(:, 1)); % Positive x values
    ref_dir1(:, 2) = abs(ref_dir1(:, 2)); % Positive y values
else
    ref_dir1(:, 1) = abs(ref_dir1(:, 1)); % Positive x values
    ref_dir1(:, 2) = -abs(ref_dir1(:, 2)); % Negative y values
end

c = cross(t, ref_dir1, 2);
theta = atan2d(sqrt(dot(c, c, 2)), dot(t, ref_dir1, 2)); % Angle theta - relative to vertical

% Define second reference direction
ref_dir2 = zeros(size(t));
ref_dir2(:, 2) = 1;

c = cross(t, ref_dir2, 2);
phi = atan2d(sqrt(dot(c, c, 2)), dot(t, ref_dir2, 2));  % Angle phi - relative to horizontal

save(pointcloud_space_curve_filename);

% Output orientation point cloud
header = {'x', 'y', 'z', 'fibre ID', 'theta', 'phi'};
orientation_point_cloud = [point_cloud(:, 5:7), point_cloud(:,2), theta, phi];
matrix = [header; num2cell(orientation_point_cloud)];
writecell(matrix, orientation_output_filename);

toc
