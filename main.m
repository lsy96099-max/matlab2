function h = main(options)
% ===== Master function =====

%% ================= USER OPTIONS =================
options = struct();
options.n_robots_3d = 10;
options.n_robots_2d = 3;
options.n_targets   = 10;
options.env_size    = [100 100];
options.seed        = 0;
options.v_max       = 2;
options.animate     = true;
options.tMax        = 100;
options.mu_b        = 1;

set_path();

%% ================= INIT PARAMS =================
h.params = environ_constants(options);
if h.params.sim.movie
    h.params.sim.animate = 1;
end

bbox   = h.params.env.bbox;
width  = bbox(3) - bbox(1);
height = bbox(4) - bbox(2);

h.server = init_server(h);

%% ================= INIT ROBOTS =================
% ---------- 3D robots ----------
z0 = h.params.controller3d.elevation_min;
q0 = bsxfun(@plus, bbox(1:2)' + [0.4*width; 0], ...
    bsxfun(@times, [0.2*width; 0.1*height], rand(2, h.params.env.nBots_3d)));
q0 = [q0; z0*ones(1, h.params.env.nBots_3d)];

% ---------- 2D robots ----------
z1 = 0;
q1 = bsxfun(@plus, bbox(1:2)' + [0.4*width; 0], ...
    bsxfun(@times, [0.2*width; 0.1*height], rand(2, h.params.env.nBots_2d)));
q1 = [q1; z1*ones(1, h.params.env.nBots_2d)];

[V, C] = VoronoiBounded([q0, q1]', h.params.env.bndry);

for i = 1:h.params.env.nBots_3d
    h.robots(i) = init_robot(h, q0(:,i), V(C{i},:), i, true);
end

for i = 1:h.params.env.nBots_2d
    idx = i + h.params.env.nBots_3d;
    h.robots(idx) = init_robot(h, q1(:,i), V(C{idx},:), i, false);
end

%% ================= INIT TARGETS (FIXED) =================
% 约定：
% occ_grid == 0 → 可通行
% occ_grid == 1 → 障碍

occ = h.params.env.occ_grid;

% ✅ 只在“可通行栅格”生成目标
[free_y, free_x] = find(occ == 0);

nFree = length(free_x);
assert(nFree >= h.params.env.nTargets, ...
    'Free space too small for targets!');

perm = randperm(nFree, h.params.env.nTargets);

t0 = zeros(3, h.params.env.nTargets);
for i = 1:h.params.env.nTargets
    ix = free_x(perm(i));
    iy = free_y(perm(i));

    % 栅格 → 世界坐标
    t0(1,i) = (ix - 0.5) * h.params.env.grid_res;
    t0(2,i) = (iy - 0.5) * h.params.env.grid_res;
    t0(3,i) = 2*pi*rand;
end

for i = 1:h.params.env.nTargets
    h.targets(i) = init_target(h, t0(:,i), i);
end

%% ================= INIT ANIMATION =================
if h.params.sim.animate > 0
    h = init_animation(h);
end

%% ================= RUN =================
try
    parpool;
catch
end

h = run_simulation(h);

%% ================= CLEAN =================
if h.params.sim.animate == 1
    h = rmfield(h, 'viz');
end

names = fieldnames(h.robots);
names = names(~strcmp(names,'data'));
h.robots = rmfield(h.robots,names);

names = fieldnames(h.server);
names = names(~strcmp(names,'data'));
h.server = rmfield(h.server,names);

end
