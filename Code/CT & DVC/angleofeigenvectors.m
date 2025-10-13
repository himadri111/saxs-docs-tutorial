%% Difference between principle strain direction and fibre orientation

straindata = readmatrix("167208_tissuestrain-sw75.Lstr.csv");
fibreorientationdata = readmatrix("167208_orientation.csv");
outputfilename = '167208_strain_direction.csv';

%reorder strain file to same order as orientation

% Extract the x, y, z coordinates from both files
coords1 = fibreorientationdata(:, 1:3);
coords2 = straindata(:, 2:4);  
  
% Define tolerance for floating-point comparison
tolerance = 1e-2;  % Set this to an appropriate value based on your data

% Use knnsearch to find the nearest neighbors in coords2 for each row in coords1
match_idx = knnsearch(coords2, coords1);

% Initialize an array to hold the reordered data
reordered_straindata = zeros(size(straindata));

% Loop over each row in file1 and find the approximately matching row in file2
for i = 1:size(coords1, 1)
     % Compute the distance between the matched row and the current row
    dist = norm(coords1(i, :) - coords2(match_idx(i), :));

    % Check if the closest match is within the tolerance
    if dist <= tolerance
        reordered_straindata(i, :) = straindata(match_idx(i), :);
    else
        warning('No matching coordinates found for row %d in file1 within tolerance.', i);
        disp(dist);
        reordered_straindata(i, :) = straindata(match_idx(i), :);
    end
end

disp('Reordering complete with tolerance.');

exx = reordered_straindata(:,10);
eyy = reordered_straindata(:,11);
ezz = reordered_straindata(:,12);
exy = reordered_straindata(:,13);
eyz = reordered_straindata(:,14);
exz = reordered_straindata(:,15);

theta_v = fibreorientationdata(:,5); % fibre orientation with vertical
phi_h = fibreorientationdata(:,6); % fibre orientation with horizontal

% convert angles to radians
theta_v = deg2rad(theta_v);
phi_h = deg2rad(phi_h);

% Assuming the strain tensor components are stored in arrays:
% exx, eyy, ezz, exy, exz, eyz, each of size (N x 1).

% Initialize arrays for fibre line vector L, eigenvalues, eigenvectors, and angles
N = length(exx); % Number of points
L = zeros(N,3); % to store Lx, Ly, Lz
eigenvalues = zeros(N, 3); % To store eigenvalues
eigenvectors = zeros(N, 3, 3); % To store eigenvectors [point, eigenvector, component]
angles = zeros(N, 3); % To store angles with fibre

% Loop through each point and compute eigenvalues/vectors and angles
for i = 1:N
    % Calculate line direction vector (Lx, Ly, Lz)
    L(i,1) = cos(theta_v(i)) * sin(phi_h(i));
    L(i,2) = sin(theta_v(i)) * sin(phi_h(i));
    L(i,3) = cos(phi_h(i));
    % normalise L
    L(i,:) = L(i,:)/norm(L(i,:));

    % Form the strain tensor for the current point
    strain_tensor = [exx(i), exy(i), exz(i);
                     exy(i), eyy(i), eyz(i);
                     exz(i), eyz(i), ezz(i)];
    
    % Compute eigenvalues and eigenvectors
    [V, D] = eig(strain_tensor);
    
    % Store eigenvalues and eigenvectors
    eigenvalues(i, :) = diag(D)';
    eigenvectors(i, :, :) = V;
    
    % Compute angles with fibre
    for j = 1:3 % Loop over the three eigenvectors
        
        cos_alpha = dot(V(:,j), transpose(L(i,:))); % Dot product with line direction
        angles(i,j) = acos(cos_alpha); % Angle in radians
        if angles(i,j) > (pi/2)
            angles(i,j) = pi-angles(i,j);
        end

    end
end

% Convert angles from radians to degrees (optional)
angles = rad2deg(angles);

header = {'x', 'y', 'z', 'ep1_alpha', 'ep3_alpha'};
straindirdiff_file = [coords1 angles(:,3) angles(:,1)];
matrix = [header; num2cell(straindirdiff_file)];
writecell(matrix, outputfilename);

