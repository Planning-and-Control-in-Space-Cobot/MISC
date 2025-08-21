function plot_scene_export_pretty(matFile, varargin)
% plot_scene_export_pretty('scene_export.mat', 'decim', 2)
% Pretty 3D plot with no edges, soft lighting, orthographic view,
% pruned (blue) and unpruned (red) paths, and robot boxes.

% -------- options --------
p = inputParser;
addParameter(p, 'decim', 1, @(x) isnumeric(x) && isscalar(x) && x>=1);
parse(p, varargin{:});
DECIM = round(p.Results.decim);

if nargin < 1 || isempty(matFile), matFile = 'scene_export.mat'; end
S = load(matFile);

% -------- colors --------
prunedColor   = [0.00, 0.30, 1.00];  % blue
unprunedColor = [1.00, 0.10, 0.10];  % red
envFace       = 0.7*[1 1 1];        % light gray

% -------- data ----------
env_points             = S.env_points;
voxel_points           = S.voxel_points;
voxel_quads            = S.voxel_quads;
voxel_tris             = S.voxel_tris;

pruned_positions       = S.pruned_positions;
pruned_orientations    = S.pruned_orientations;
unpruned_positions     = S.unpruned_positions;
unpruned_orientations  = S.unpruned_orientations;

robot_size             = S.robot_size(:)';        
boundsX = S.boundsX(:)'; boundsY = S.boundsY(:)'; boundsZ = S.boundsZ(:)';

% -------- figure/axes --------
%set(0,'DefaultFigureRenderer','painters'); 


fig = figure('Color','w', 'Renderer','opengl',  ...
             'Units','normalized', 'Position',[0.05 0.05 0.9 0.85]); % large
set(fig, 'GraphicsSmoothing', 'on');


ax = axes('Parent', fig); hold(ax, 'on');
set(ax, 'FontSize', 16)
axis(ax, 'equal'); box(ax, 'on'); grid(ax, 'on');
set(ax, 'Projection', 'orthographic');
xlabel(ax,'X - m'); ylabel(ax,'Y - m'); zlabel(ax,'Z - m');
xlim(ax, boundsX); ylim(ax, boundsY); zlim(ax, boundsZ);
view(ax, [-65 20]);      
axis(ax, 'vis3d');      

% -------- environment --------
if ~isempty(voxel_points) && (size(voxel_quads,1) > 0 || size(voxel_tris,1) > 0)
    V = voxel_points;
    F = [];
    if ~isempty(voxel_quads)
        Q = voxel_quads + 1;
        F = [F; Q(:, [1 2 3]); Q(:, [1 3 4])]; 
    end
    if ~isempty(voxel_tris)
        T = voxel_tris + 1;
        F = [F; T]; 
    end
    hp_env = patch('Vertices', V, 'Faces', F, ...
        'FaceColor', envFace, 'EdgeColor', 'none', ...
        'FaceAlpha', 1.0, 'Parent', ax, 'HandleVisibility','off');
    material(hp_env, 'dull');
else
    if ~isempty(env_points)
        scatter3(ax, env_points(:,1), env_points(:,2), env_points(:,3), ...
            1, envFace, 'filled', 'MarkerEdgeAlpha',1, 'MarkerFaceAlpha',1, ...
            'HandleVisibility','off');
    end
end

% -------- paths --------
h_legend = gobjects(0);
if size(pruned_positions,1) >= 2
    h1 = plot3(ax, pruned_positions(:,1), pruned_positions(:,2), pruned_positions(:,3), ...
        '-', 'Color', prunedColor, 'LineWidth', 2.0, 'DisplayName','Pruned path');
    h_legend(end+1) = h1; %#ok<AGROW>
end
% if size(unpruned_positions,1) >= 2
%     h2 = plot3(ax, unpruned_positions(:,1), unpruned_positions(:,2), unpruned_positions(:,3), ...
%         '-', 'Color', unprunedColor, 'LineWidth', 2.0, 'DisplayName','Unpruned path');
%     h_legend(end+1) = h2; %#ok<AGROW>
% end

% -------- robot boxes --------
[boxV, boxF] = box_vertices_and_faces(robot_size(1), robot_size(2), robot_size(3));
for i = 1:DECIM:size(pruned_positions,1)
    p = pruned_positions(i, :).';
    q = pruned_orientations(i, :).';
    Rm = quat_to_rotm(q);
    Vt = (Rm * boxV.').'+ p.';
    hp = patch('Vertices', Vt, 'Faces', boxF, ...
        'FaceColor', prunedColor, 'EdgeColor', 'none', ...
        'FaceAlpha', 0.65, 'Parent', ax, 'HandleVisibility','off');
    material(hp, 'dull');
end
% for i = 1:DECIM:size(unpruned_positions,1)
%     p = unpruned_positions(i, :).';
%     q = unpruned_orientations(i, :).';
%     Rm = quat_to_rotm(q);
%     Vt = (Rm * boxV.').'+ p.';
%     hp = patch('Vertices', Vt, 'Faces', boxF, ...
%         'FaceColor', unprunedColor, 'EdgeColor', 'none', ...
%         'FaceAlpha', 0.55, 'Parent', ax, 'HandleVisibility','off');
%     material(hp, 'dull');
% end

% -------- lights & legend --------
camlight(ax, 'headlight');
camlight(ax, 'left');
lighting(ax, 'flat');
% title(ax, 'Environment with Pruned (blue) and Unpruned (red) paths');

if ~isempty(h_legend)
    % legend(ax, h_legend, 'Location','northeast');
    % legend(ax, h_legend, 'Location', 'best', 'Box', 'off');

end

% -------- maximize use of space --------
axis tight;
set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02)); % reduce margins
end

% ===== helpers =====
function [V, F] = box_vertices_and_faces(dx, dy, dz)
hx = dx/2; hy = dy/2; hz = dz/2;
V = [ -hx,-hy,-hz;
       hx,-hy,-hz;
       hx, hy,-hz;
      -hx, hy,-hz;
      -hx,-hy, hz;
       hx,-hy, hz;
       hx, hy, hz;
      -hx, hy, hz ];
F = [ 1 2 3 4;
      5 6 7 8;
      1 2 6 5;
      3 4 8 7;
      2 3 7 6;
      1 4 8 5 ];
end

function Rm = quat_to_rotm(q)
x=q(1); y=q(2); z=q(3); w=q(4);
xx=x*x; yy=y*y; zz=z*z;
xy=x*y; xz=x*z; yz=y*z;
wx=w*x; wy=w*y; wz=w*z;
Rm = [ 1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy);
       2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx);
       2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy) ];
end
