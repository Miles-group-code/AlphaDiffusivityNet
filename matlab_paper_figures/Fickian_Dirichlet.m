%% --- Fickian Model with Dirichlet Boundary: Figure 5 ---
% Point-source weak (C^0) doppelganger. A point source b0*delta(x-z) forces a
% jump in u', so the Fickian flux freedom delta(D u') = C1 - db0*H(x-z) can only
% stay continuous if the source strength also changes (db0 ~= 0). The resulting
% delta D = (C1 - db0 H)/u' is continuous but KINKS at z: distinct (D1,b0_1) and
% (D2,b0_2) produce the same density u. C^1-at-z (Assumption) excludes the kink.
% Absorbing (homogeneous) Dirichlet walls u(0)=u(1)=0: u peaks at the source, so
% u' flips sign across z but stays nonzero on each side, and delta D = g/u' is
% finite throughout (the single jump node z is filled by the one-sided slope).
clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

N  = 4001;
x  = linspace(0,1,N)';
dx = x(2)-x(1);
z  = 0.5;  [~, iz] = min(abs(x - z));

gamma = 5;            % decay > 0
b0_1  = 10;           % true point-source strength
db0   = 4;            % source perturbation
b0_2  = b0_1 + db0;   % phantom strength (b0_1 ~= b0_2)

D1 = 2 + 0.5*cos(4*pi*x);

% --- True solution: (D1 u')' - gamma u + b0_1 delta(x-z) = 0, u(0)=u(1)=0 ---
u1  = fick_solve(D1, b0_1, gamma, 0, 0, iz, dx, N);
u1p = gradient(u1, dx);
u1p(iz) = (u1(iz) - u1(iz-1)) / dx;   % one-sided at the source jump node

% --- Continuous-at-z flux constant: C1 = db0 * u1'(z^-) * D1(z) / b0_1 ---
C1 = db0 * u1p(iz) * D1(iz) / b0_1;

% --- delta D = (C1 - db0 H(x-z)) / u1' ; D2 = D1 + delta D ---
g  = C1 - db0*(x > z);
dD = g ./ u1p;
D2 = D1 + dD;

u2 = fick_solve(D2, b0_2, gamma, 0, 0, iz, dx, N);

%% --- Verification ---
fprintf('max|u1-u2| = %.2e  (%.4f%% of peak)\n', max(abs(u1-u2)), max(abs(u1-u2))/max(u1)*100);
fprintf('min(D2) = %.4f   min|u1''| off-source = %.4f (u'' ~= 0 on each side)\n', ...
        min(D2), min(abs(u1p([2:iz-3 iz+3:end-1]))));
dDp = gradient(dD, dx);
fprintf('[dD''](z): slope %.3f -> %.3f  (C^1 kink at source)\n', mean(dDp(iz-5:iz-2)), mean(dDp(iz+2:iz+5)));

%% --- Plot ---
figure('Color','w','Position',[100 100 1500 450]);

ax1 = subplot(1,3,1);
plot(x, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax2 = subplot(1,3,2);
plot(x, D1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, D2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax3 = subplot(1,3,3);
stem(z, b0_1, '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(z, b0_2, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',7, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Point sources','FontWeight','normal');
legend({sprintf('$b_0^{(1)} = %g$',b0_1), sprintf('$b_0^{(2)} = %g$',b0_2)}, 'Interpreter','latex','Location','northeast');
xlim([0 1]); ylim([0 b0_2*1.5]);

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

exportgraphics(gcf, 'Fickian_Dirichlet.pdf', 'ContentType', 'vector');

%% --- Fickian conservative FDM: (D u')' - gamma u + b0 delta(x-z) = 0, Dirichlet ---
function u = fick_solve(D, b0, gamma, uL, uR, iz, dx, N)
  Dp = (D + [D(2:end); D(end)])/2;   Dm = ([D(1); D(1:end-1)] + D)/2;
  main = -(Dp + Dm)/dx^2 - gamma;    lo = Dm(2:end)/dx^2;   up = Dp(1:end-1)/dx^2;
  M = spdiags([[lo; 0] main [0; up]], [-1 0 1], N, N);
  rhs = zeros(N,1);  rhs(iz) = -b0/dx;
  M(1,:) = 0; M(1,1) = 1; rhs(1) = uL;
  M(N,:) = 0; M(N,N) = 1; rhs(N) = uR;
  u = M \ rhs;
end
