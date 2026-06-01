%% --- Ito Model with unknown Robin Boundary (weak doppelganger) ---
clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2;                     % uniform line width

L = 1; N = 1000;
xmesh = linspace(0, L, N); dx = xmesh(2) - xmesh(1);
kappa1 = 2;       
kappa_L1 = 2;     

x0 = 0.5; [~, idx_source] = min(abs(xmesh - x0));

D1 = 2 + 0.5*sin(2*pi*xmesh);
b1_mag = 10;

flux1 = zeros(size(xmesh));
flux1(1:idx_source) = -b1_mag/2;
flux1(idx_source+1:end) = b1_mag/2;

ratio  = 1.5;
b2_mag = b1_mag * ratio;          
flux2  = flux1  * ratio;
D2     = D1     * ratio;          
kappa2   = kappa1   * ratio;      
kappa_L2 = kappa_L1 * ratio;

u1 = zeros(size(xmesh));

u1(1) = abs(flux1(1)) / kappa1;
for i = 2:idx_source
    u1(i) = u1(i-1) - (dx / D1(i)) * flux1(i);
end
for i = idx_source+1:N
    u1(i) = u1(i-1) - (dx / D1(i)) * flux1(i);
end


u2 = zeros(size(xmesh));
u2(1) = abs(flux2(1)) / kappa2;
for i = 2:idx_source
    u2(i) = u2(i-1) - (dx / D2(i)) * flux2(i);
end
for i = idx_source+1:N
    u2(i) = u2(i-1) - (dx / D2(i)) * flux2(i);
end

gamma = 0.1;   % decay rate 
MB1 = gamma * trapz(xmesh, u1) + kappa1*u1(1)   + kappa_L1*u1(end);
MB2 = gamma * trapz(xmesh, u2) + kappa2*u2(1)   + kappa_L2*u2(end);
fprintf('Mass balance check — b1: %.4f (should be %.1f)\n', MB1, b1_mag);
fprintf('Mass balance check — b2: %.4f (should be %.1f)\n', MB2, b2_mag);
%% --- Plot ---
figure('Color', 'w', 'Position', [100, 100, 1500, 400]);

ax1 = subplot(1,4,1);
plot(xmesh, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(xmesh, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax2 = subplot(1,4,2);
plot(xmesh, D1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(xmesh, D2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','best');

ax3 = subplot(1,4,3);
stem(x0, b1_mag, '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(x0, b2_mag, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',7, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Point sources','FontWeight','normal');
legend({sprintf('$b_0^{(1)} = %g$',b1_mag), sprintf('$b_0^{(2)} = %g$',b2_mag)}, 'Interpreter','latex','Location','best');
xlim([0 1]); ylim([0 b2_mag*1.5]);

ax4 = subplot(1,4,4);
bvals = [kappa1 kappa2; kappa_L1 kappa_L2];
bh = bar(bvals, 'grouped');
bh(1).FaceColor = c1; bh(1).EdgeColor = 'none';
bh(2).FaceColor = c2; bh(2).EdgeColor = 'none';
set(ax4,'XTickLabel',{'$\kappa_0$','$\kappa_L$'},'TickLabelInterpreter','latex');
ylabel('$\kappa$','Interpreter','latex'); title('Robin permeabilities','FontWeight','normal'); ylim([0 4]);

axs = [ax1 ax2 ax3 ax4]; tags = {'(a)','(b)','(c)','(d)'};
for ii = 1:numel(axs)
    a = axs(ii);
    box(a,'off'); grid(a,'on');
    set(a,'FontSize',11,'LineWidth',0.8,'GridAlpha',0.15);
    a.Title.FontWeight = 'normal'; a.Title.Units = 'normalized'; a.Title.Position(1:2) = [0.5 1.04];
    yl = ylim(a); ylim(a, [yl(1), yl(2)+0.12*(yl(2)-yl(1))]);
    text(a, 0.035, 0.96, tags{ii}, 'Units','normalized','Interpreter','tex', ...
         'FontWeight','bold','FontSize',12,'VerticalAlignment','top');
end
set(findall(gcf,'Type','legend'),'FontSize',9,'Box','off');

exportgraphics(gcf, 'Ito_Unknown_Robin.pdf', 'ContentType', 'vector');