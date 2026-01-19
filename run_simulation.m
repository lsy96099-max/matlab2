function h = run_simulation(h)

% number of robots
nr_2d = h.params.env.nBots_2d;
nr_3d = h.params.env.nBots_3d;
nr = nr_2d + nr_3d;

R = @(t) [cos(t) -sin(t); sin(t) cos(t)];

x0 = min(h.params.env.bndry(:,1));
x1 = max(h.params.env.bndry(:,1));
y0 = min(h.params.env.bndry(:,2));
y1 = max(h.params.env.bndry(:,2));

m = min([h.params.env.rate, h.params.sensor2d.rate, h.params.controller2d.rate]);
h.params.sim.rate = lcm(h.params.env.rate/m, lcm(h.params.sensor2d.rate/m, h.params.controller2d.rate/m)) * m;

% Initialize necessary fields for robots
for i = 1:nr
    if ~isfield(h.robots(i), 'q')
        warning('Robot %d is missing q field, initialized to default position', i);
        h.robots(i).q = [0; 0; 1]; % Default position and height
    end
    if ~isfield(h.robots(i), 'voronoi') || isempty(h.robots(i).voronoi)
        h.robots(i).voronoi = [0 0; 0 0; 0 0; 0 0]; % Initialize to a simple rectangular region
    end
    if ~isfield(h.robots(i), 'temp')
        h.robots(i).temp = struct();
    end
    h.robots(i).temp.voronoi_old = h.robots(i).voronoi;
end

h = log_data(h, 0);

% Run the simulation
iter = 0; % counter for iterations
t = 1; % counter for measurements

if h.params.sim.animate == 0
    h.viz.t0 = CTimeleft(h.params.sim.tMax);
    h.viz.t0.timeleft();
end

done = false;
while ~done
    % if get measurements
    if mod(iter, h.params.sim.rate / h.params.sensor2d.rate) == 0
        % compute Voronoi regions
        % Extract all robot positions for Voronoi region computation
        robot_positions = zeros(nr, 2);
        for i = 1:nr
            robot_positions(i, :) = h.robots(i).q(1:2)';
        end
        
        [V, C] = VoronoiBounded(robot_positions, h.params.env.bndry); 
        
        for i = 1:nr
            h.robots(i).temp.voronoi_old = h.robots(i).voronoi; % save old cell
            % Safety check: ensure C{i} contains valid indices
            if ~isempty(C{i})
                h.robots(i).voronoi = V(C{i},:); % extract Voronoi region for robot
            else
                warning('Robot %d has an empty Voronoi region, using default value', i);
                % Use a default region around the robot
                robot_pos = h.robots(i).q(1:2);
                radius = 10; % Default radius
                % Create a square region centered on the robot
                h.robots(i).voronoi = [
                    robot_pos(1)-radius, robot_pos(2)-radius;
                    robot_pos(1)+radius, robot_pos(2)-radius;
                    robot_pos(1)+radius, robot_pos(2)+radius;
                    robot_pos(1)-radius, robot_pos(2)+radius
                ];
            end
        end
        
        % get measurements
        qt = [h.targets.q];
        qt = qt(:, [h.targets.active]); % ignore inactive targets
        for i = 1:nr
            h.robots(i) = get_measurements(h.robots(i), qt, t);
        end
        
        if h.params.sim.decentralized
            % exchange particles
            h = exchange_particles(h);
        
            % phd prediction
            h = phd_filter_prediction(h);
        
            % phd update
            h = phd_filter_update(h);
        end
        
        % run centralized filter for comparison
        h = phd_filter_server(h);
        if h.params.sim.decentralized
            check_filters(h);
            % note: filters are different when order of measurements is different
        else
            r = h.robots;
            x = h.server.state.x;
            w = h.server.state.w;
            parfor i = 1:length(r)
                inpol = inpolygon(x(1,:), x(2,:), r(i).voronoi(:,1), r(i).voronoi(:,2));
                r(i).state.x = find(inpol);
                r(i).state.w = w(inpol);
            end
            h.robots = r;
        end
        
        % log data
        h = log_data(h, t);
        
        for i = 1:nr
%             t_ind = max(1, t-3) : t+1;
%             if mean(h.robots(i).data.term_criterion(t_ind)) < h.params.sim.entropy_thresh
%                 h.robots(i).active = false;
%             else
%                 h.robots(i).active = true;
%             end
            h.robots(i).data.active(t) = h.robots(i).active;
        end
        
        if h.server.data.term_criterion(t+1) < h.params.sim.entropy_thresh
            h.server.active = false;
        else
            h.server.active = true;
        end
        h.server.data.active(t) = h.server.active;
    end
   
    % find next action
    if mod(iter, h.params.sim.rate / h.params.controller2d.rate) == 0
        for i = 1:nr_3d
            if h.robots(i).active
                x = h.server.state.x(:, h.robots(i).state.x);
                if h.robots(i).controller.use_phd
                    w_3d = h.robots(i).state.w;
                else
                    w_3d = ones(1, size(x,2));
                end
                w_total = sum(w_3d);
                cent = x * w_3d' / w_total; % compute weighted centroid of region
                covar = bsxfun(@times, (x - cent), w_3d) * (x - cent)' / w_total;
                if isnan(cent) % happens when no particles in voronoi cell
                    cent = h.robots(i).q(1:2);
                    r_target = h.robots(i).q(3)^2 * h.robots(i).sensor.radius;
                else
                    D = eig(covar);
                    r_target = 3*sqrt(max(D));
                end
                % Cover the cell if no targets visible
                r_cell = (polycirc(h.robots(i).voronoi, cent, 0) + ...
                    polycirc(h.robots(i).voronoi, cent, 1))/2; % average of inscribed and circumscribed radii
            else
                w_total = 0;
                cent = h.robots(i).q(1:2);
                % Cover the cell if no targets visible
                r_target = h.robots(i).q(3)^2 * h.robots(i).sensor.radius;
                r_cell = r_target;
            end
            r = (h.robots(i).controller.w_cell * r_cell + w_total * r_target) / ...
                (h.robots(i).controller.w_cell + w_total);
%             fprintf('rc:%02.2f rt:%02.2f t:%02.2f\n', r_cell, r_target, r);
            h.robots(i).temp.goal = [cent; 
                min(max(r / h.robots(i).sensor.radius, ...
                    h.robots(i).controller.elevation_min), ...
                    h.robots(i).controller.elevation_max)];           
            h.robots(i).data.path{t} = h.robots(i).temp.goal;
        end

        for i = 1:nr_2d
            if h.robots(i+nr_3d).active
                x = h.server.state.x(:, h.robots(i+nr_3d).state.x);
                if h.robots(i+nr_3d).controller.use_phd
                    w_2d = h.robots(i+nr_3d).state.w;
                else
                    w_2d = ones(1, size(x,2));
                end
                cent = x * w_2d' / sum(w_2d); % compute weighted centroid of region
                if isnan(cent) % happens when no particles in voronoi cell
                    cent = h.robots(i+nr_3d).q(1:2);
                end
            else
                cent = h.robots(i+nr_3d).q(1:2);
            end
            h.robots(i+nr_3d).temp.goal = [cent;0];
            h.robots(i+nr_3d).data.path{t} = h.robots(i+nr_3d).temp.goal;
        end  
    end
    
    % move targets
   % ================= MOVE TARGETS (OBSTACLE SAFE) =================
grid = h.params.env.occ_grid;        % 0 = free, 1 = obstacle
res  = h.params.env.grid_res;
[H,W] = size(grid);

for i = 1:length(h.targets)
    if ~h.targets(i).active
        continue;
    end

    q = h.targets(i).q;

    % ===== 1. proposed motion =====
    step = h.params.target.v_max * rand / h.params.sim.rate;
    dq   = R(q(3)) * step * [1;0];
    q_new = q;
    q_new(1:2) = q(1:2) + dq;
    q_new(3)   = q(3) + h.params.target.w_max * randn / h.params.sim.rate;

    % ===== 2. world → grid =====
    ix = floor(q_new(1)/res) + 1;
    iy = floor(q_new(2)/res) + 1;

    inside = ix>=1 && ix<=W && iy>=1 && iy<=H;

    % ===== 3. collision check =====
    if inside && grid(iy,ix) == 0
        % ✅ free cell → accept move
        q = q_new;
    else
        % ❌ obstacle / out → turn away, stay put
        q(3) = q(3) + pi/2 + pi*rand;   % random turn
    end

    % ===== 4. update =====
    h.targets(i).q = q;
    h.targets(i).data.q(:,t+1) = q;
end

    
    % add targets
    nt = poissrnd(h.params.phd.mu_b); % number of births
    for i = 1:nt
        n = length(h.targets);
        h.targets(n+1) = init_target(h, draw_target_location(h), n+1);
        h.targets(n+1).data.t_birth = t;
    end

    % move robots
    for i = 1:nr_3d
        dist = norm(h.robots(i).temp.goal - h.robots(i).q);
        dist_action = h.robots(i).controller.vel / h.params.sim.rate;
        if dist < dist_action
            h.robots(i).q = h.robots(i).temp.goal;
        else
            h.robots(i).q = h.robots(i).q + (h.robots(i).temp.goal - h.robots(i).q) / dist * dist_action;
        end
    end
    
    for i = 1:nr_2d
        dist = norm(h.robots(i+nr_3d).temp.goal - h.robots(i+nr_3d).q);
        dist_action = h.robots(i+nr_3d).controller.vel / h.params.sim.rate;
        if dist < dist_action
            h.robots(i+nr_3d).q = h.robots(i+nr_3d).temp.goal;
        else
            h.robots(i+nr_3d).q = h.robots(i+nr_3d).q + (h.robots(i+nr_3d).temp.goal - h.robots(i+nr_3d).q) / dist * dist_action;
        end
    end

    % draw latest estimate
    if h.params.sim.animate > 0
        h = animate(h, t);
    end
    
    done = false;
    if t == h.params.sim.tMax
        done = true;
    end
    
    iter = iter+1;
    if mod(iter, h.params.sim.rate / h.params.sensor2d.rate) == 0
        t = t+1;
        if ~h.params.sim.animate
            h.viz.t0.timeleft();
        end
    end
    
    if done
        break;
    end
end

end
