% plot_pcds_scientific.m
% Scientific plots for three PCDs with Start/End poses.
% Single-color point clouds (no gradient). Other features preserved:
% cropping, downsampling, start/goal boxes, clean styling.
% Start/End boxes are now OPAQUE (no transparency).

clear; clc; rng(0);

%% SIMPLE MAP
simple.file = 'simpleMap.pcd';
simple.boundsMin = [-5, 0, -2.5];
simple.boundsMax = [ 4, 8,  2.5];
simple.startPos  = [ 3.5, 0.5, 2.0];
simple.goalPos   = [ 3.0, 7.0, 0.0];
simple.startQuat = [0, 0, 0, 1];
simple.goalQuat  = [0, 0, 0, 1];

%% INTERMEDIARY MAP — downsample to 1/10
middle.file = 'middleMap.pcd';
middle.boundsMin = [1.5, 1.0, 0.0];
middle.boundsMax = [4.0, 6.0, 10.0];
middle.startPos  = [2.75, 2.0, 1.0];
middle.goalPos   = [2.75, 2.0, 7.0];
middle.startQuat = [0, 0.707, 0, 0.707];
middle.goalQuat  = [0, 0.707, 0, 0.707];

%% COMPLEX MAP
complexM.file = 'complexMap.pcd';
complexM.boundsMin = [0.0, 3.0, 0.0];
complexM.boundsMax = [3.0, 6.5, 7.0];
complexM.startPos  = [0.5, 3.5, 1.0];
complexM.goalPos   = [0.5, 5.0, 6.0];
complexM.startQuat = [0, 0, 0, 1];
complexM.goalQuat  = [0, 0, 0, 1];

%% ONE-SPHERE MAP (sphere only) — file produced by the Python script above
% NOTE: If MATLAB errors on '.pcl', rename to 'oneSphere.pcd' or set oneSphere.file = 'oneSphere.pcd'.
oneSphere.file = 'oneSphere.pcd';
% Movement along X: start (0,0,5) -> goal (10,0,5), sphere centered at (5,0,5) with radius 2.5
oneSphere.boundsMin = [-1.0, -3.0, 0.0];
oneSphere.boundsMax = [11.0,  3.0, 10.0];
oneSphere.startPos  = [0.0, 0.0, 5.0];
oneSphere.goalPos   = [10.0, 0.0, 5.0];
oneSphere.startQuat = [0, 0, 0, 1];
oneSphere.goalQuat  = [0, 0, 0, 1];

%% Style
ptSize       = 5;
robotLWH     = [0.45, 0.45, 0.12];

% Colors
startColor   = [0.10 0.55 0.30];   % green-ish for Start
endColor     = [0.80 0.20 0.20];   % red-ish for End
pointColor   = [0.20 0.20 0.80];   % bluish for point clouds

% Opaque Start/End boxes (no transparency)
faceAlpha    = 1.0;
edgeAlpha    = 1.0;

fontSize     = 12;

%% Plot
plotOne_scientific(simple,   'Simple Map',       ptSize, robotLWH, startColor, endColor, faceAlpha, edgeAlpha, fontSize, ...
                   'PointColor', pointColor);

plotOne_scientific(middle,   'Intermediary Map', ptSize, robotLWH, startColor, endColor, faceAlpha, edgeAlpha, fontSize, ...
                   'KeepRatio', 0.8, ...
                   'Crop', [0.2, 0.1], ...
                   'PointColor', pointColor);   % [zMin, yMargin]

plotOne_scientific(complexM, 'Complex Map',      ptSize, robotLWH, startColor, endColor, faceAlpha, edgeAlpha, fontSize, ...
                   'PointColor', pointColor);

% NEW: One-Sphere map
plotOne_scientific(oneSphere, 'One Sphere',      ptSize, robotLWH, startColor, endColor, faceAlpha, edgeAlpha, fontSize, ...
                   'PointColor', pointColor);

fprintf('Done.\n');

%% ======================== FUNCTIONS ========================

function plotOne_scientific(cfg, titleStr, ptSize, robotLWH, startColor, endColor, faceAlpha, edgeAlpha, fontSize, varargin)
    % Defaults
    opts = struct( ...
        'KeepRatio', 1.0, ...
        'Crop',      [], ...             % [zMin, yMargin]
        'PointColor',[0.2 0.2 0.8] ...   % fixed RGB color for cloud
    );
    opts = parseNV(opts, varargin{:});

    fpath = fullfile(pwd, cfg.file);
    if ~isfile(fpath)
        warning('Missing file: %s (skipping)', cfg.file);
        return;
    end

    % Read PCD/PLY
    pt  = pcread(fpath);
    XYZ = pt.Location;
    XYZ = XYZ(all(isfinite(XYZ),2), :);

    % Downsample
    if opts.KeepRatio < 1 && ~isempty(XYZ)
        n  = size(XYZ,1);
        k  = max(1, round(n*opts.KeepRatio));
        idx = randperm(n, k);
        XYZ = XYZ (idx,:);
        fprintf('Downsampled "%s": kept %d of %d points (%.0f%%).\n', ...
            cfg.file, size(XYZ,1), n, opts.KeepRatio*100);
    end

    % Crop (if requested)
    if ~isempty(opts.Crop) && ~isempty(XYZ)
        zMin    = opts.Crop(1);
        yMargin = opts.Crop(2);
        yMax    = max(XYZ(:,2));
        mask    = XYZ(:,3) >= zMin & XYZ(:,2) <= (yMax - yMargin);
        before  = size(XYZ,1);
        XYZ     = XYZ(mask,:);
        fprintf('Cropped "%s": kept %d of %d points (Z>=%.2f, Y<=%.2f).\n', ...
            cfg.file, size(XYZ,1), before, zMin, yMax - yMargin);
    end

    % === Plot ===
    f = figure('Color','w','Name',cfg.file); %#ok<NASGU>
    ax = axes('Parent', gcf); hold(ax,'on');

    if ~isempty(XYZ)
        scatter3(ax, XYZ(:,1), XYZ(:,2), XYZ(:,3), ptSize, ...
            repmat(opts.PointColor, size(XYZ,1),1), 'filled', 'MarkerFaceAlpha', 0.9);
        xlim(ax, [min(XYZ(:,1)) max(XYZ(:,1))]);
        ylim(ax, [min(XYZ(:,2)) max(XYZ(:,2))]);
        zlim(ax, [min(XYZ(:,3)) max(XYZ(:,3))]);
    end

    % Bounds box (semi-transparent face for context)
    addBoundsBox(ax, cfg.boundsMin, cfg.boundsMax, [0.6 0.6 0.6], 0.05, 0.5);

    % Start/End robots — OPAQUE
    drawRobotBoxPose(ax, cfg.startPos, cfg.startQuat, robotLWH, startColor, 1.0, 1.0, 'Start');
    drawRobotBoxPose(ax, cfg.goalPos,  cfg.goalQuat,  robotLWH, endColor,   1.0, 1.0, 'End');

    % Axes cosmetics
    xlabel(ax,'X [m]'); ylabel(ax,'Y [m]'); zlabel(ax,'Z [m]');
    title(ax, sprintf('%s', titleStr), 'Interpreter','none', 'FontWeight','bold');
    axis(ax,'equal'); grid(ax,'on'); box(ax,'on');
    ax.GridAlpha = 0.15; ax.LineWidth = 0.8;
    view(ax, 35, 25); rotate3d(ax,'on');
    set(ax,'FontSize', fontSize);
    plot3(ax, 0,0,0, '.', 'Color',[0.2 0.2 0.2]); % origin
end

function s = parseNV(s, varargin)
    if mod(numel(varargin),2)~=0
        error('Name–value arguments must come in pairs.');
    end
    for k = 1:2:numel(varargin)
        name = char(varargin{k});
        val  = varargin{k+1};
        if isfield(s, name)
            s.(name) = val;
        else
            error('Unknown option "%s".', name);
        end
    end
end

function addBoundsBox(ax, bmin, bmax, colorRGB, alphaFace, lw)
    [F,V] = boundsBoxFaces(bmin, bmax);
    patch('Faces',F,'Vertices',V, ...
          'FaceColor',colorRGB, 'FaceAlpha',alphaFace, ...
          'EdgeColor',colorRGB, 'EdgeAlpha',0.6, ...
          'LineWidth',lw, 'Parent',ax);
end

function [F,V] = boundsBoxFaces(bmin, bmax)
    x0=bmin(1); y0=bmin(2); z0=bmin(3);
    x1=bmax(1); y1=bmax(2); z1=bmax(3);
    V = [x0 y0 z0; x1 y0 z0; x1 y1 z0; x0 y1 z0;  % bottom
         x0 y0 z1; x1 y0 z1; x1 y1 z1; x0 y1 z1]; % top
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
end

function drawRobotBoxPose(ax, pos, quat_xyzw, sizeLWH, colorRGB, faceAlpha, edgeAlpha, labelStr)
    L = sizeLWH(1); W = sizeLWH(2); H = sizeLWH(3);
    vLocal = [ ...
        -L/2 -W/2 -H/2;
         L/2 -W/2 -H/2;
         L/2  W/2 -H/2;
        -L/2  W/2 -H/2;
        -L/2 -W/2  H/2;
         L/2 -W/2  H/2;
         L/2  W/2  H/2;
        -L/2  W/2  H/2];
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];

    Rm = quat2rotm_local(quat_xyzw);
    vWorld = (Rm * vLocal.').';
    vWorld = vWorld + pos(:).';

    patch('Faces',F,'Vertices',vWorld, ...
          'FaceColor',colorRGB, 'FaceAlpha',faceAlpha, ...
          'EdgeColor',colorRGB, 'EdgeAlpha',edgeAlpha, ...
          'LineWidth',1.0, 'Parent',ax);

    % Heading indicator
    fwdLocal = [L/2, 0, 0]; tailLocal = [0, 0, 0];
    fwdWorld  = (Rm * fwdLocal.').';
    tailWorld = (Rm * tailLocal.').';
    plot3(ax, [tailWorld(1)+pos(1), fwdWorld(1)+pos(1)], ...
             [tailWorld(2)+pos(2), fwdWorld(2)+pos(2)], ...
             [tailWorld(3)+pos(3), fwdWorld(3)+pos(3)], ...
             '-', 'Color', colorRGB, 'LineWidth', 1.5);

    text(ax, pos(1), pos(2), pos(3) + H*0.75, labelStr, ...
         'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
         'FontWeight','bold', 'Color', colorRGB);
end

function Rm = quat2rotm_local(qxyzw)
    x = qxyzw(1); y = qxyzw(2); z = qxyzw(3); w = qxyzw(4);
    n = sqrt(x*x + y*y + z*z + w*w);
    if n > 0, x=x/n; y=y/n; z=z/n; w=w/n; end
    xx = x*x; yy = y*y; zz = z*z; xy = x*y; xz = x*z; yz = y*z; wx = w*x; wy = w*y; wz = w*z;
    Rm = [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy);
              2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx);
              2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)];
end
