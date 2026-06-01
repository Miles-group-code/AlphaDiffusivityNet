%% --- Fickian Model with Robin Boundary: Figure 6 ---
% Point-source weak (C^0) doppelganger at an UNKNOWN-permeability Robin wall.
% As in the Dirichlet case the point source forces delta(D u') = C1 - db0*H(x-z)
% and a joint (D, b0) change; here the wall permeabilities kappa absorb the flux
% offset, so the triple (D, b0, kappa) is non-identifiable. The perturbation
% delta D = (C1 - db0 H)/u' kinks at z and is excluded by C^1-at-z (Assumption).
% Robin form: -D u'(0) + k0 u(0) = q0 ; D u'(1) + k1 u(1) = q1.
clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

N  = 4001;
x  = linspace(0,1,N)';
dx = x(2)-x(1);
z  = 0.5;  [~, iz] = min(abs(x - z));

gamma = 5;             % decay > 0
b0_1  = 10;            % true point-source strength
db0   = 5;             % source perturbation
b0_2  = b0_1 + db0;    % phantom strength
k0_1  = 2;  k1_1 = 2;  % true Robin permeabilities
q0    = 0;  q1   = 0;  % homogeneous (leaky) Robin -D u' + k u = 0, exterior at 0

D1 = 2 + 0.5*sin(2*pi*x);

% --- True solution: u peaks at the source; u' nonzero on each side ---
u1  = robin_solve(D1, b0_1, gamma, k0_1, k1_1, q0, q1, iz, dx, N);
u1p = gradient(u1, dx);
u1p(iz) = (u1(iz) - u1(iz-1)) / dx;   % one-sided at the source jump node

% --- Continuous-at-z flux constant ---
C1 = db0 * u1p(iz) * D1(iz) / b0_1;

% --- delta D and the compensating permeabilities (k2 = k1 -/+ flux offset / u(wall)) ---
g  = C1 - db0*(x > z);
dD = g ./ u1p;
D2 = D1 + dD;
k0_2 = k0_1 + C1 / u1(1);
k1_2 = k1_1 - (C1 - db0) / u1(N);

u2 = robin_solve(D2, b0_2, gamma, k0_2, k1_2, q0, q1, iz, dx, N);

%% --- Verification ---
fprintf('max|u1-u2| = %.2e  (%.4f%% of peak)\n', max(abs(u1-u2)), max(abs(u1-u2))/max(u1)*100);
fprintf('min(D2) = %.4f   min|u1''| off-source = %.4f (u'' ~= 0 on each side)\n', ...
        min(D2), min(abs(u1p([2:iz-3 iz+3:end-1]))));
fprintf('b0: %g -> %g   k0: %.3f -> %.3f   k1: %.3f -> %.3f\n', b0_1, b0_2, k0_1, k0_2, k1_1, k1_2);

%% --- Plot ---
figure('Color','w','Position',[80 80 1500 450]);

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
stem(z, b0_1, '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(z, b0_2, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',7, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Point sources','FontWeight','normal');
legend({sprintf('$b_0^{(1)} = %g$',b0_1), sprintf('$b_0^{(2)} = %g$',b0_2)}, 'Interpreter','latex','Location','northeast');
xlim([0 1]); ylim([0 b0_2*1.5]);

ax4 = subplot(1,4,4);
bh = bar([k0_1 k0_2; k1_1 k1_2]);
bh(1).FaceColor = c1; bh(1).EdgeColor = 'none';
bh(2).FaceColor = c2; bh(2).EdgeColor = 'none';
set(ax4,'XTickLabel',{'$\kappa_0$','$\kappa_L$'},'TickLabelInterpreter','latex');
ylabel('$\kappa$','Interpreter','latex'); title('Robin permeabilities','FontWeight','normal');

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

exportgraphics(gcf, 'FickianRobin.pdf', 'ContentType', 'vector');

%% --- Fickian conservative FDM with Robin walls ---
% -D u'(0) + k0 u(0) = q0 ;  D u'(1) + k1 u(1) = q1  (2nd-order one-sided)
function u = robin_solve(D, b0, gamma, k0, k1, q0, q1, iz, dx, N)
  Dp = (D + [D(2:end); D(end)])/2;   Dm = ([D(1); D(1:end-1)] + D)/2;
  main = -(Dp + Dm)/dx^2 - gamma;    lo = Dm(2:end)/dx^2;   up = Dp(1:end-1)/dx^2;
  M = spdiags([[lo; 0] main [0; up]], [-1 0 1], N, N);
  rhs = zeros(N,1);  rhs(iz) = -b0/dx;
  M(1,:) = 0; M(1,1) = 3*D(1)/(2*dx) + k0; M(1,2) = -4*D(1)/(2*dx); M(1,3) = D(1)/(2*dx); rhs(1) = q0;
  M(N,:) = 0; M(N,N) = 3*D(N)/(2*dx) + k1; M(N,N-1) = -4*D(N)/(2*dx); M(N,N-2) = D(N)/(2*dx); rhs(N) = q1;
  u = M \ rhs;
end
