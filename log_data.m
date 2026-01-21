function h = log_data(h, t)

targets = [h.targets.q];
targets = targets(:,[h.targets.active]);

% server
% 检查h.server结构中是否存在state字段
if isfield(h.server, 'state') && isfield(h.server.state, 'lambda')
    % expected number of targets in voronoi cell
    h.server.data.lambda(t+1) = h.server.state.lambda;
    % entropy
    lambda = h.server.state.lambda;
    w = h.server.state.w/lambda;
    w = w(w>0); % ignore particles with zero weight
    h.server.data.entropy(t+1) = lambda*(1 - log(lambda) - w*log(w.'));
    assert(~isnan(h.server.data.entropy(t+1)))
    % termination criterion
    h.server.data.term_criterion(t+1) = (h.server.data.entropy(t+1) - lambda) / lambda;
else
    % 如果state字段不存在，设置默认值
    h.server.data.lambda(t+1) = 0;
    h.server.data.entropy(t+1) = 0;
    h.server.data.term_criterion(t+1) = 0;
end

% true number of targets in voronoi cell
inpol = inpolygon(targets(1,:), targets(2,:), h.params.env.bndry(:,1), h.params.env.bndry(:,2));
h.server.data.lambda_true(t+1) = sum(inpol);

if t > 0 && isfield(h.server, 'state') && isfield(h.server.state, 'x') && isfield(h.server.state, 'w')
    % create image of phd
    bbox = h.params.env.bbox;
    origin = bbox(1:2)';
    sz = fliplr(h.params.env.size);
    idx = particle2idx(h.server.state.x, origin, h.params.phd.grid_size); % index in local map
    w_grid = accumarray(idx, h.server.state.w, sz);

    % find local maxima in phd
    % 使用imregionalmax替代vision.LocalMaximaFinder
    bw = imregionalmax(w_grid);
    % 应用阈值过滤
    threshold = 0.05;
    bw = bw & (w_grid > threshold);
    
    % 使用regionprops获取局部最大值的位置
    stats = regionprops(bw, w_grid, 'PixelIdxList', 'PixelValues', 'WeightedCentroid');
    
    % 按强度排序并限制数量
    max_num = ceil(2*h.server.state.lambda);
    if length(stats) > max_num
        [~, idx] = sort(cellfun(@max, {stats.PixelValues}), 'descend');
        stats = stats(idx(1:max_num));
    end
    
    % 获取局部最大值位置
    if ~isempty(stats)
        location = zeros(length(stats), 2);
        for i = 1:length(stats)
            [row, col] = ind2sub(size(w_grid), stats(i).PixelIdxList(1));
            location(i,:) = [col, row];
        end
    else
        location = zeros(0, 2);
    end
    
    % compute ospa metric
   ind_phd = sub2ind(h.params.env.size, location(:,1), location(:,2));

% ====== 兜底修复：防止索引超过粒子数量 ======
Np = size(h.server.state.x, 2);
ind_phd = ind_phd(ind_phd >= 1 & ind_phd <= Np);

if isempty(ind_phd)
    h.server.data.ospa(t) = h.params.sim.ospa_cutoff;
    h.server.data.num_targets_found(t) = 0;
    h.server.data.num_targets_matched(t) = 0;
    return;
end
% ===============================================

est_targets = h.server.state.x(:, ind_phd);

    dist = pdist2(targets(1:2,:)', est_targets');
    cutoff = h.params.sim.ospa_cutoff;
    dist(dist > cutoff) = cutoff;
    [assign, ospa] = munkres(dist);
    h.server.data.ospa(t) = (ospa + cutoff * abs(diff(size(dist)))) / max(size(dist));
    h.server.data.num_targets_found(t) = size(location,1);
    costs = dist(assign);
    h.server.data.num_targets_matched(t) = sum(costs(:) < cutoff);
else
    % 如果state字段不存在或t=0，设置默认值
    if t > 0
        h.server.data.ospa(t) = 0;
        h.server.data.num_targets_found(t) = 0;
        h.server.data.num_targets_matched(t) = 0;
    end
    location = [];
    ind_phd = [];
end

% robots
for i = 1:length(h.robots)
    % expected number of targets in voronoi cell
    if isfield(h.robots(i), 'state') && isfield(h.robots(i).state, 'lambda')
        h.robots(i).data.lambda(t+1) = h.robots(i).state.lambda;
    else
        h.robots(i).data.lambda(t+1) = 0;
    end
    
    % true number of targets in voronoi cell
    if isfield(h.robots(i), 'voronoi') && ~isempty(h.robots(i).voronoi) && size(h.robots(i).voronoi, 2) >= 2
        inpol = inpolygon(targets(1,:), targets(2,:), ...
            h.robots(i).voronoi(:,1), h.robots(i).voronoi(:,2));
        h.robots(i).data.lambda_true(t+1) = sum(inpol);
    else
        h.robots(i).data.lambda_true(t+1) = 0;
    end
    
    % expected number of targets in footprint
    if isfield(h.server, 'state') && isfield(h.server.state, 'x') && ...
       isfield(h.robots(i), 'state') && isfield(h.robots(i).state, 'x')
        x = h.server.state.x(:, h.robots(i).state.x);
        
        if isfield(h.robots(i), 'q') && isfield(h.robots(i), 'sensor') && ...
           isfield(h.robots(i).sensor, 'footprint')
            if h.robots(i).q(3) > 0
                footprint = bsxfun(@plus, h.robots(i).q(1:2)', h.robots(i).q(3)*h.robots(i).sensor.footprint);
            else
                footprint = bsxfun(@plus, h.robots(i).q(1:2)', h.robots(i).sensor.footprint);
            end
            
            inpol = inpolygon(x(1,:), x(2,:), footprint(:,1), footprint(:,2));
            if isfield(h.robots(i).state, 'w')
                h.robots(i).data.lambda_foot(t+1) = sum(h.robots(i).state.w(inpol));
            else
                h.robots(i).data.lambda_foot(t+1) = 0;
            end
            
            % true number of targets in footprint
            inpol = inpolygon(targets(1,:), targets(2,:), footprint(:,1), footprint(:,2));
            h.robots(i).data.lambda_foot_true(t+1) = sum(inpol);
        else
            h.robots(i).data.lambda_foot(t+1) = 0;
            h.robots(i).data.lambda_foot_true(t+1) = 0;
        end
    else
        h.robots(i).data.lambda_foot(t+1) = 0;
        h.robots(i).data.lambda_foot_true(t+1) = 0;
    end
    
    % entropy
    if isfield(h.robots(i), 'state') && isfield(h.robots(i).state, 'lambda') && ...
       isfield(h.robots(i).state, 'w')
        lambda = h.robots(i).state.lambda;
        w = h.robots(i).state.w;
        w = w(w>0); % ignore particles with zero weight
        if isempty(w)
            h.robots(i).data.entropy(t+1) = 0;
        else
            h.robots(i).data.entropy(t+1) = lambda - w*log(w.');
        end
        if ~isempty(w)
            assert(~isnan(h.robots(i).data.entropy(t+1)))
        end
        % termination criterion
        h.robots(i).data.term_criterion(t+1) = (h.robots(i).data.entropy(t+1) - lambda) / max(0.5,lambda);
    else
        h.robots(i).data.entropy(t+1) = 0;
        h.robots(i).data.term_criterion(t+1) = 0;
    end
    
    % pose
    if isfield(h.robots(i), 'q')
        h.robots(i).data.q(:,t+1) = h.robots(i).q;
    end
    
    if t > 0
        % measurement
        if isfield(h.robots(i), 'z')
            h.robots(i).data.z{t} = h.robots(i).z;
        end
        
        % ospa metric - 只有在有必要的数据时才计算
        if exist('ind_phd', 'var') && ~isempty(ind_phd) && ...
           isfield(h.server, 'state') && isfield(h.server.state, 'x') && ...
           isfield(h.robots(i), 'state') && isfield(h.robots(i).state, 'x') && ...
           isfield(h.robots(i), 'voronoi') && ~isempty(h.robots(i).voronoi) && size(h.robots(i).voronoi, 2) >= 2
            inpol = inpolygon(targets(1,:), targets(2,:), ...
                h.robots(i).voronoi(:,1), h.robots(i).voronoi(:,2));
            tar = targets(1:2,inpol);
            inpol = ismember(ind_phd, h.robots(i).state.x);
            est = h.server.state.x(:,ind_phd(inpol));
            if ~isempty(tar) && ~isempty(est)
                dist = pdist2(tar', est');
                dist(dist > cutoff) = cutoff;
                [assign, ospa] = munkres(dist);
                h.robots(i).data.ospa(t) = ospa + cutoff * abs(diff(size(dist)));
                h.robots(i).data.num_targets_found(t) = sum(inpol);
                costs = dist(assign);
                h.robots(i).data.num_targets_matched(t) = sum(costs(:) < cutoff);
            else
                h.robots(i).data.ospa(t) = 0;
                h.robots(i).data.num_targets_found(t) = 0;
                h.robots(i).data.num_targets_matched(t) = 0;
            end
        else
            h.robots(i).data.ospa(t) = 0;
            h.robots(i).data.num_targets_found(t) = 0;
            h.robots(i).data.num_targets_matched(t) = 0;
        end
    end
end
