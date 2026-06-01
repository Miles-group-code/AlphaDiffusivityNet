%% --- Ito Model with UNKNOWN Robin Boundary (weak doppelganger) ---
% Same-source degeneracy: with delta b0 = 0 the Ito ambiguity W = delta D * u
% satisfies W'' = 0 (no corner at z), so W = c(x - z). Choosing W(z) = 0 makes
% delta D = c(x - z)/u genuinely C^1 at the source -- it is NOT the excluded
% kinked C/u mode. The two UNKNOWN wall permeabilities absorb the flux offset
% (delta kappa_0 = c/u(0), delta kappa_L = -c/u(L)), so (D, b0, kappa) is
% non-identifiable even within the C^1 class and even with b0 held fixed.
% A KNOWN permeability would force c = 0 (identifiable, like Neumann).
clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

L = 1; N = 4001;
x = linspace(0, L, N)'; dx = x(2) - x(1);
z = 0.5; [~, iz] = min(abs(x - z));

gamma  = 5;            % decay > 0
b0     = 10;           % point-source strength (SHARED by both models)
k0_1   = 2; kL_1 = 2;  % true Robin permeabilities
D1     = 2 + 0.5*sin(2*pi*x);

% --- True solution: Ito Robin BVP  (D u)'' - gamma u = -b0 delta(x-z) ---
u1 = ito_robin(D1, b0, gamma, k0_1, kL_1, iz, dx, N);

% --- Same-source doppelganger: W = c (x - z),  delta D = W/u1 (C^1 at z) ---
c    = 0.6;                       % flux offset (small enough: D2>0, kappa>0)
W    = c*(x - z);
dD   = W ./ u1;
D2   = D1 + dD;
b0_2 = b0;                        % SAME source strength
k0_2 = k0_1 + c/u1(1);            % delta kappa_0 = W'(0)/u(0)
kL_2 = kL_1 - c/u1(N);            % delta kappa_L = -W'(L)/u(L)

u2 = ito_robin(D2, b0_2, gamma, k0_2, kL_2, iz, dx, N);

%% --- Verification ---
fprintf('max|u1-u2| = %.2e  (%.5f%% of peak)\n', max(abs(u1-u2)), max(abs(u1-u2))/max(u1)*100);
fprintf('min(D2) = %.4f (must be > 0)\n', min(D2));
fprintf('b0: %g -> %g (identical)   kappa0: %.3f -> %.3f   kappaL: %.3f -> %.3f\n', ...
        b0, b0_2, k0_1, k0_2, kL_1, kL_2);
MB1 = gamma*trapz(x,u1) + k0_1*u1(1) + kL_1*u1(N);
MB2 = gamma*trapz(x,u2) + k0_2*u2(1) + kL_2*u2(N);
fprintf('mass balance   b1: %.4f   b2: %.4f   (both should be %.1f)\n', MB1, MB2, b0);

%% --- Plot ---
figure('Color', 'w', 'Position', [100, 100, 1500, 450]);

ax1 = subplot(1,4,1);
plot(x, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax2 = subplot(1,4,2);
plot(x, D1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, D2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax3 = subplot(1,4,3);
stem(z, b0,   '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(z, b0_2, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',7, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Identical sources','FontWeight','normal');
legend({sprintf('$b_0^{(1)} = %g$',b0), sprintf('$b_0^{(2)} = %g$',b0_2)}, 'Interpreter','latex','Location','northeast');
xlim([0 1]); ylim([0 b0*1.5]);

ax4 = subplot(1,4,4);
bvals = [k0_1 k0_2; kL_1 kL_2];
bh = bar(bvals, 'grouped');
bh(1).FaceColor = c1; bh(1).EdgeColor = 'none';
bh(2).FaceColor = c2; bh(2).EdgeColor = 'none';
set(ax4,'XTickLabel',{'$\kappa_0$','$\kappa_L$'},'TickLabelInterpreter','latex');
ylabel('$\kappa$','Interpreter','latex'); title('Robin permeabilities','FontWeight','normal'); ylim([0 4]);
legend([bh(1) bh(2)], {'Model 1','Model 2'}, 'Interpreter','tex','Location','northeast');

axs = [ax1 ax2 ax3 ax4]; tags = {'(a)','(b)','(c)','(d)'};
for ii = 1:numel(axs)
    a = axs(ii);
    box(a,'off'); grid(a,'on');
    set(a,'FontSize',11,'LineWidth',1.2,'GridAlpha',0.15,'GridLineWidth',0.5,'TickDir','out','TickLength',[0.015 0.025]);
    a.TitleFontSizeMultiplier = 1.45; a.LabelFontSizeMultiplier = 1.7;
    p = a.Position; a.Position = [p(1) p(2) p(3) p(4)*0.90];   % top headroom so titles aren't clipped
    a.Title.FontWeight = 'normal'; a.Title.Units = 'normalized'; a.Title.Position(1:2) = [0.5 1.03];
    yl = ylim(a); ylim(a, [yl(1), yl(2)+0.12*(yl(2)-yl(1))]);
    text(a, 0.035, 0.95, tags{ii}, 'Units','normalized','Interpreter','tex', ...
         'FontWeight','bold','FontSize',18,'VerticalAlignment','top');
end
set(findall(gcf,'Type','legend'),'FontSize',16,'Box','off');
ax4.XAxis.FontSize = 18;   % kappa_0, kappa_L are category labels, not tick values

exportgraphics(gcf, 'Ito_Unknown_Robin.pdf', 'ContentType', 'vector');

%% --- Ito Robin FDM:  (D u)'' - gamma u = -b0 delta(x-z);  -dn(Du) + k u = 0 ---
function u = ito_robin(D, b0, gamma, k0, kL, iz, dx, N)
  e = ones(N,1);
  A = spdiags([e -2*e e], [-1 0 1], N, N)/dx^2;   % second difference
  M = A*spdiags(D,0,N,N) - gamma*speye(N);          % (D u)'' - gamma u
  rhs = zeros(N,1); rhs(iz) = -b0/dx;
  % Robin rows on the Ito flux (D u):  -dn(Du) + k u = 0  (2nd-order one-sided)
  M(1,:) = 0;
  M(1,1) = 3*D(1)/(2*dx) + k0; M(1,2) = -4*D(2)/(2*dx); M(1,3) = D(3)/(2*dx); rhs(1) = 0;
  M(N,:) = 0;
  M(N,N) = 3*D(N)/(2*dx) + kL; M(N,N-1) = -4*D(N-1)/(2*dx); M(N,N-2) = D(N-2)/(2*dx); rhs(N) = 0;
  u = M\rhs;
end
