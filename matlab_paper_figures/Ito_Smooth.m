%% --- Ito Model with Smooth Birth Source (Non-identifiable) Figure 2 ---

clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

d = 10; % decay rate

D1     = @(x) 1.5 + sin(4*x);
D1p    = @(x) 4*cos(4*x);
D1pp   = @(x) -16*sin(4*x);

b1 = @(x) 1.0 * (1./(0.1*sqrt(2*pi))) .* exp(-0.5*((x - 0.5)./0.1).^2);

N = 1e4;
xmesh = linspace(0,1,N);

odefun1 = @(x,y) [ ...
    y(2); ...
    ( d*y(1) - b1(x) - 2*D1p(x).*y(2) - D1pp(x).*y(1) ) ./ D1(x) ...
];

bcfun   = @(ya,yb) [ya(1); yb(1)]; % Dirichlet: u(0)=0, u(1)=0
sol1    = bvp4c(odefun1, bcfun, bvpinit(xmesh, @(x)[x*(1-x); 0]));
y1      = deval(sol1, xmesh);
u1      = y1(1,:)'; % u(x)
u1p     = y1(2,:)'; % u'(x)

C  = 1.0;             
b2 = @(x) b1(x) + C;

% With Q := D2*u, we need Q'' = d*u - b2
b2_vals = b2(xmesh)';
F   = d*u1 - b2_vals;                    
Phi1 = cumtrapz(xmesh, F); % integral of F
Phi2 = cumtrapz(xmesh, Phi1); % double integral of F

% Choose constants so Q(0)=0 and Q(1)=0
K0 = 0;
K1 = -Phi2(end);
Q  = Phi2 + K1*xmesh' + K0;

eps_reg = 1e-12;
D2 = Q ./ max(abs(u1), eps_reg);

m = 5; 
D2(1:m)      = D2(m+1) + (xmesh(1:m) - xmesh(m+1)) * ((D2(m+2)-D2(m+1))/(xmesh(m+2)-xmesh(m+1)));
D2(end-m+1:end) = D2(end-m) + (xmesh(end-m+1:end) - xmesh(end-m)) * ((D2(end-m)-D2(end-m-1))/(xmesh(end-m)-xmesh(end-m-1)));

h = xmesh(2) - xmesh(1);
D2p  = gradient(D2, h);
D2pp = gradient(D2p, h);

D2i   = @(x) interp1(xmesh, D2,   x, 'pchip', 'extrap');
D2pi  = @(x) interp1(xmesh, D2p,  x, 'pchip', 'extrap');
D2ppi = @(x) interp1(xmesh, D2pp, x, 'pchip', 'extrap');


odefun2 = @(x,y) [ ...
    y(2); ...
    ( d*y(1) - b2(x) - 2*D2pi(x).*y(2) - D2ppi(x).*y(1) ) ./ D2i(x) ...
];

sol2 = bvp4c(odefun2, bcfun, bvpinit(xmesh, @(x) deval(sol1,x)));
y2  = deval(sol2, xmesh);
u2  = y2(1,:)';
%% --- Plot ---
diff_norm = norm(u1 - u2);
fprintf('||u1 - u2||_2 = %.3e\n', diff_norm);
figure('Color', 'w', 'Position', [100, 100, 1500, 450]);
ax1 = subplot(1,3,1);
plot(xmesh, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(xmesh, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax2 = subplot(1,3,2);
plot(xmesh, D1(xmesh), '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(xmesh, D2,        '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax3 = subplot(1,3,3);
plot(xmesh, b1(xmesh), '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(xmesh, b2(xmesh), '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b(x)$','Interpreter','latex'); title('Source terms','FontWeight','normal');
legend({'$b_1(x)$','$b_2(x)$'}, 'Interpreter','latex','Location','northeast');

axs = [ax1 ax2 ax3]; tags = {'(a)','(b)','(c)'};
for ii = 1:numel(axs)
    a = axs(ii);
    box(a,'off'); grid(a,'on');
    set(a,'FontSize',11,'LineWidth',1.2,'GridAlpha',0.15,'GridLineWidth',0.5,'TickDir','out','TickLength',[0.015 0.025],'TickLabelInterpreter','tex');
    a.TitleFontSizeMultiplier = 1.45; a.LabelFontSizeMultiplier = 1.7;
    p = a.Position; a.Position = [p(1) p(2) p(3) p(4)*0.90];   % top headroom so titles aren't clipped
    a.Title.FontWeight = 'normal'; a.Title.Units = 'normalized'; a.Title.Position(1:2) = [0.5 1.03];
    yl = ylim(a); ylim(a, [yl(1), yl(2)+0.12*(yl(2)-yl(1))]);
    text(a, 0.035, 0.95, tags{ii}, 'Units','normalized','Interpreter','tex', ...
         'FontWeight','bold','FontSize',18,'VerticalAlignment','top');
end
set(findall(gcf,'Type','legend'),'FontSize',16,'Box','off');

exportgraphics(gcf, 'Ito_Smooth.pdf', 'ContentType', 'vector');
