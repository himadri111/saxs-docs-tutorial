%% TomoSAXS metrics and loaded orientation calculation
%% Finds fibre tangent values, curvature, displacements and strain along 
% fibres, and calculates orientation of fibres after deformation
% INPUTS: reference point_spacecurve file output from TomoSAXS_orientation_optimised 
%         .disp file from idvc 
%          
% OUTPUTS: fibre metrics .csv 'x', 'y', 'z', 'theta', 'phi', 'delta theta', 'delta phi', 'k', 'delta k', 'm', 'L', 'Lsmooth'
%          pointcloud mask of loaded fibres .csv  'x', 'y', 'z'
%          loaded fibre orientation .csv 'x', 'y', 'z', 'fibre ID', 'theta', 'phi'
%          
tic
%% Import point cloud, displacements and space curve information
pointcloud_spacecurve = "167208_pointcloud_spacecurve.mat";
dispfile = readmatrix("1627208_displacement.disp",'FileType','text');

output_filename_metrics = "162708_fibre_metrics.csv";
output_filename_orientation = "162709_orientation.csv";
pointcloud_mask_output_filename = "167209_pointcloud_mask.csv";

load(pointcloud_spacecurve) %reference point cloud 

%reorder disp file to same order as pointcloud

% Extract the x, y, z coordinates from both files
coords1 = point_cloud(:, 5:7);  
coords2 = dispfile(:, 2:4);  

% Define tolerance for floating-point comparison
tolerance = 1e-2;  % Set this to an appropriate value based on your data

% Use knnsearch to find the nearest neighbors in coords2 for each row in coords1
match_idx = knnsearch(coords2, coords1);

% Initialize an array to hold the reordered data
reordered_dispfile = zeros(size(dispfile));

% Loop over each row in file1 and find the approximately matching row in file2
for i = 1:size(coords1, 1)
     % Compute the distance between the matched row and the current row
    dist = norm(coords1(i, :) - coords2(match_idx(i), :));

    % Check if the closest match is within the tolerance
    if dist <= tolerance
        reordered_dispfile(i, :) = dispfile(match_idx(i), :);
    else
        warning('No matching coordinates found for row %d in file1 within tolerance.', i);
        disp(dist);
        reordered_dispfile(i, :) = dispfile(match_idx(i), :);
    end
end

disp('Reordering complete with tolerance.');

if isempty(dispfile)
    u=[]; v=[]; w=[];
else 

x = reordered_dispfile(:,2);
y = reordered_dispfile(:,3);
z = reordered_dispfile(:,4);     
objmin1 = reordered_dispfile(:,6);
u = reordered_dispfile(:,7);
v = reordered_dispfile(:,8);
w = reordered_dispfile(:,9);

point_cloud_new = [point_cloud(:,1:4) (point_cloud(:,5)+u) (point_cloud(:,6)+v) (point_cloud(:,7)+w) point_cloud(:,8:10)];

% Number of fibers
fibers_idx_new = unique(point_cloud_new(:, 2));  % Extract unique fiber indices
nfibres_new = length(fibers_idx_new);            % Number of fibers

%% seed points every voxel for mask
inc_mask = 1;

% Preallocate estimated space for point_cloud
point_cloud_mask = zeros(nfibres_new * 10000, 10);  % Estimate large enough initial space
pc_count = 0;  % Keep track of actual number of points

for i = 1:nfibres_new
    
    % Extract data for the current fiber
    fiber_id_new = fibers_idx_new(i);
    fiber_idx_new = (point_cloud_new(:, 2) == fiber_id_new);
    fiber_data_new = point_cloud_new(fiber_idx_new, :);
    n = size(fiber_data_new, 1); % Number of points in the fiber

    % Preallocate fibre coordinates for current fibre
    fibre = fiber_data_new(:, 5:7);        % Extract coordinates for fibre
    
    %% Interpolate points along fibre
    pos = fibre(1, :);              % Start with the first point
    p1 = fibre(2, :);               % Second point
    dir = (p1 - pos) / norm(p1 - pos);  % Initial direction vector

    for p = 2:n-1
        p0 = pos(end, :);
        dist = norm(p1 - p0);       % Distance between latest stored and current point

        while inc_mask < dist
            % Calculate direction and new point
            dir_new = (p1 - p0) / norm(p1 - p0);
            p_new = p0 + inc_mask * dir_new;  % Create new point at distance 'inc_mask'
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
    point_cloud_mask(pc_count + 1:pc_count + num_pos, :) = fpc;
    pc_count = pc_count + num_pos;  % Update total point count

    % Reset variables for the next iteration
    pos = [];
    dir = [];
end

% Trim point_cloud to actual size
point_cloud_mask = point_cloud_mask(1:pc_count, :);

% Write to file
writematrix(point_cloud_mask(:, 5:7), pointcloud_mask_output_filename);

%% space curve fitting - fit 3rd order polynomial to data
% Number of fibers
fibers_idx_new = unique(point_cloud_new(:, 2));  % Extract unique fiber indices
nfibres_new = length(fibers_idx_new);            % Number of fibers

% Preallocate cell arrays and variables
fx_new = cell(nfibres_new, 1);
fy_new = cell(nfibres_new, 1);
fz_new = cell(nfibres_new, 1);
s_linear_new = [];

parfor i = 1:nfibres_new
    % Extract data for the current fiber
    fiber_id_new = fibers_idx_new(i);
    fiber_idx_new = (point_cloud_new(:, 2) == fiber_id_new);
    fiber_data_new = point_cloud_new(fiber_idx_new, :);
    npts_new = size(fiber_data_new, 1); % Number of points in the fiber

    if npts_new < 4
        % Assign empty values for fibers with fewer than 4 points
        fx_new{i} = [];
        fy_new{i} = [];
        fz_new{i} = [];
        s_linear_new = [s_linear_new; NaN(npts_new, 1)];
    else
        % Calculate distances for s_0
        p0_new = fiber_data_new(1:end-1, 5:7);
        p1_new = fiber_data_new(2:end, 5:7);
        distances_new = sqrt(sum((p1_new - p0_new).^2, 2));  % Vectorized distance calculation
        s_0_new = [0; cumsum(distances_new)];            % Cumulative distance for parameter s

        % Fit the point cloud data using poly3 (degree 3 polynomial)
        fx_0_new = polyfit(s_0_new, fiber_data_new(:, 5), 3);  % Fit x-coordinates
        fy_0_new = polyfit(s_0_new, fiber_data_new(:, 6), 3);  % Fit y-coordinates
        fz_0_new = polyfit(s_0_new, fiber_data_new(:, 7), 3);  % Fit z-coordinates

        % Store results
        fx_new{i} = fx_0_new;
        fy_new{i} = fy_0_new;
        fz_new{i} = fz_0_new;

        % Append s_0 to s_linear
        s_linear_new = [s_linear_new; s_0_new];
    end
end

%% tangent calculations
t_new = [];  % Preallocate tangent matrix
parfor i = 1:nfibres_new
    npts_new = sum(point_cloud_new(:, 2) == fibers_idx_new(i)); % Number of points in fibre

    if npts_new < 4
       t_new = [t_new; NaN(npts_new, 3)];
    else
        % Retrieve coefficients for polynomials
        coeff_fx_new = fx_new{i};
        coeff_fy_new = fy_new{i};
        coeff_fz_new = fz_new{i};
        s_0_new = s_linear_new(point_cloud_new(:, 2) == fibers_idx_new(i), 1);

        % Tangent calculation
        t0_new = [3*coeff_fx_new(1)*s_0_new.^2 + 2*coeff_fx_new(2)*s_0_new + coeff_fx_new(3), ...
              3*coeff_fy_new(1)*s_0_new.^2 + 2*coeff_fy_new(2)*s_0_new + coeff_fy_new(3), ...
              3*coeff_fz_new(1)*s_0_new.^2 + 2*coeff_fz_new(2)*s_0_new + coeff_fz_new(3)];
        t_new = [t_new; t0_new];
    end
end

%% f_orientation
% Define reference direction for orientation calculation
ref_dir1_new = t_new;
ref_dir1_new(:, 3) = 0; % z=0 to create a plane

if sum(sign(t_new(:, 1)) == sign(t_new(:, 2))) / length(t_new) > 0.5
    ref_dir1_new(:, 1) = abs(ref_dir1_new(:, 1)); % Positive x values
    ref_dir1_new(:, 2) = abs(ref_dir1_new(:, 2)); % Positive y values
else
    ref_dir1_new(:, 1) = abs(ref_dir1_new(:, 1)); % Positive x values
    ref_dir1_new(:, 2) = -abs(ref_dir1_new(:, 2)); % Negative y values
end

c_new = cross(t_new, ref_dir1_new, 2);
theta_new = atan2d(sqrt(dot(c_new, c_new, 2)), dot(t_new, ref_dir1_new, 2)); % Angle theta - relative to vertical

% Define second reference direction
ref_dir2_new = zeros(size(t_new));
ref_dir2_new(:, 2) = 1;

c_new = cross(t_new, ref_dir2_new, 2);
phi_new = atan2d(sqrt(dot(c_new, c_new, 2)), dot(t_new, ref_dir2_new, 2));  % Angle phi - relative to horizontal

% Output orientation point cloud
header = {'x', 'y', 'z', 'fibre ID', 'theta', 'phi', };
orientation_point_cloud_new = [point_cloud_new(:, 5:7), point_cloud_new(:,2), theta_new, phi_new];
matrix = [header; num2cell(orientation_point_cloud_new)];
writecell(matrix, output_filename_orientation);

end 

t=[];
k=[];
k_new = [];
m=[];
L=[];
Lsmooth=[];
Lobjmin=[];

% nfibres = sum(diff(point_cloud(:,2))~=0)+1;

for i=1:nfibres

npts=length(point_cloud((point_cloud(:,2)==i),5)); %number of points in fibre

if npts<4
    L=[L; ones(npts,1)*NaN];
    m=[m; ones(npts,1)*NaN];
    k=[k; ones(npts,1)*NaN];
    k_new=[k_new; ones(npts,1)*NaN];
    t=[t; ones(npts,3)*NaN];
else

coeff_fx=(fx{i});
coeff_fy=(fy{i});
coeff_fz=(fz{i});

s_0=s_linear(point_cloud(:,2)==i,1);

%% Calculate tangent
 dfx=@(s) 3*coeff_fx(:,1)*s.^2+2*coeff_fx(:,2)*s+coeff_fx(:,3);
 dfy=@(s) 3*coeff_fy(:,1)*s.^2+2*coeff_fy(:,2)*s+coeff_fy(:,3);
 dfz=@(s) 3*coeff_fz(:,1)*s.^2+2*coeff_fz(:,2)*s+coeff_fz(:,3);

t0=[dfx(s_0), dfy(s_0), dfz(s_0)];
spacing = 0:inc:inc*size(t0,1)-1;
t=[t; t0];


%% Calculate curvature
fxd1p=[polyder(fx{i})];
fyd1p=[polyder(fy{i})];
fzd1p=[polyder(fz{i})];

fxd1 = polyval(fxd1p,s_0);
fyd1 = polyval(fyd1p,s_0);
fzd1 = polyval(fzd1p,s_0);

fxd2p=[polyder(fxd1p)];
fyd2p=[polyder(fyd1p)];
fzd2p=[polyder(fzd1p)];

fxd2 = polyval(fxd2p,s_0);
fyd2 = polyval(fyd2p,s_0);
fzd2 = polyval(fzd2p,s_0);

a=fzd2.*fyd1-fyd2.*fzd1;
b=fxd2.*fzd1-fzd2.*fxd1;
c=fyd2.*fxd1-fxd2.*fyd1;
d=fxd1.^2+fyd1.^2+fzd1.^2;

% curvature calc
k=[k;((a.^2+b.^2+c.^2).^0.5).*d.^(-3/2)];

% calculate curvature for DVC updated fibres
coeff_fx_new=(fx_new{i});
coeff_fy_new=(fy_new{i});
coeff_fz_new=(fz_new{i});

s_0_new=s_linear_new(point_cloud_new(:,2)==i,1);

%% Calculate tangent
 dfx_new=@(s) 3*coeff_fx_new(:,1)*s.^2+2*coeff_fx_new(:,2)*s+coeff_fx_new(:,3);
 dfy_new=@(s) 3*coeff_fy_new(:,1)*s.^2+2*coeff_fy_new(:,2)*s+coeff_fy_new(:,3);
 dfz_new=@(s) 3*coeff_fz_new(:,1)*s.^2+2*coeff_fz_new(:,2)*s+coeff_fz_new(:,3);

t0_new=[dfx_new(s_0_new), dfy_new(s_0_new), dfz_new(s_0_new)];
spacing_new = 0:inc:inc*size(t0_new,1)-1;
t_new=[t_new; t0_new];


%% Calculate curvature
fxd1p_new=[polyder(fx_new{i})];
fyd1p_new=[polyder(fy_new{i})];
fzd1p_new=[polyder(fz_new{i})];

fxd1_new = polyval(fxd1p_new,s_0_new);
fyd1_new = polyval(fyd1p_new,s_0_new);
fzd1_new = polyval(fzd1p_new,s_0_new);

fxd2p_new=[polyder(fxd1p_new)];
fyd2p_new=[polyder(fyd1p_new)];
fzd2p_new=[polyder(fzd1p_new)];

fxd2_new = polyval(fxd2p_new,s_0_new);
fyd2_new = polyval(fyd2p_new,s_0_new);
fzd2_new = polyval(fzd2p_new,s_0_new);

a_new=fzd2_new.*fyd1_new-fyd2_new.*fzd1_new;
b_new=fxd2_new.*fzd1_new-fzd2_new.*fxd1_new;
c_new=fyd2_new.*fxd1_new-fxd2_new.*fyd1_new;
d_new=fxd1_new.^2+fyd1_new.^2+fzd1_new.^2;

% curvature calc
k_new=[k_new;((a_new.^2+b_new.^2+c_new.^2).^0.5).*d_new.^(-3/2)];

%% Find displacements in t direction, m 
if isempty(u)
    m=[];
else 
displacement=[u(point_cloud(:,2)==i) v(point_cloud(:,2)==i) w(point_cloud(:,2)==i)];
m=[m; dot(t0,displacement,2)];
L=[L; gradient(dot(t0,displacement,2),inc)+((gradient(dot(t0,displacement,2),inc)).^2)./2];

% Step 1: Fit a polynomial to the data
degree = 2 + round(size(t0,1)*inc/320); % Degree of the polynomial (choose an appropriate degree)
if degree > 9
    degree = 9; % largest fittype is poly9
end
p = polyfit(spacing, transpose(dot(t0,displacement,2)), degree);

% Step 2: Calculate the derivative (gradient) of the polynomial
% The derivative of a polynomial is found by multiplying each coefficient by its corresponding power
dp = polyder(p);

% Step 3: Evaluate the polynomial and its derivative at the original x points
y_fit = polyval(p, spacing); % Evaluate the polynomial fit at x points
gradient_at_x = polyval(dp, spacing); % Evaluate the derivative at x points
Lsmooth = [Lsmooth; transpose(gradient_at_x + (gradient_at_x.^2)./2)];

end 

end %if npts>4

end %loop through fibres

%calculate change in orientation and curvature
delta_theta = theta_new - theta;
delta_phi = phi_new - phi;
delta_k = k_new - k;

% Output curvature, strain point cloud
header = {'x', 'y', 'z', 'theta', 'phi', 'delta theta', 'delta phi', 'k', 'delta k', 'm', 'L', 'Lsmooth'};
metrics_file = [x y z theta phi delta_theta delta_phi k delta_k m L Lsmooth];
matrix = [header; num2cell(metrics_file)];
writecell(matrix, output_filename_metrics);

toc
% end