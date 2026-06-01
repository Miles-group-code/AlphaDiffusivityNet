%% ---Itô With Neumann Boundary (Weak Doppelgänger) — Figure 3 ---

clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

L  = 1;
N  = 2001;
x  = linspace(0, L, N)';
dx = x(2) - x(1);

z   = 0.5;                            
[~, iz] = min(abs(x - z));

b0    = 10;                           
gamma = 5;                            % decay > 0
C_neu = 1.5;                          % doppelgänger constant δD = C/u

% D1 = 2 + 0.8*sin(10*pi*x) + 0.5*cos(4*pi*x);
D1 = 2 + 0.8*sin(2*pi*x);

Aw = zeros(N, N);
for i = 2:N-1
    Aw(i,i-1) =  1/dx^2;
    Aw(i,i  ) = -2/dx^2;
    Aw(i,i+1) =  1/dx^2;
end

function M = apply_neumann(Aw, D, gamma, N, dx)
    M = Aw * diag(D) - gamma * eye(N);
    M(1,1) = -3*D(1)/(2*dx); M(1,2) = 4*D(2)/(2*dx); M(1,3) = -D(3)/(2*dx);
    M(N,N) =  3*D(N)/(2*dx); M(N,N-1) = -4*D(N-1)/(2*dx); M(N,N-2) = D(N-2)/(2*dx);
end

%% --- RHS: -b0·δ(x-z) ---
rhs = zeros(N,1);
rhs(iz) = -b0 / dx;
rhs(1)  = 0;   % Neumann BC 
rhs(N)  = 0;

M1  = apply_neumann(Aw, D1, gamma, N, dx);
u1  = M1 \ rhs;

%% ---D2 = D1 + C/u1  (δb0 = 0) ---
D2  = D1 + C_neu ./ u1;              
M2  = apply_neumann(Aw, D2, gamma, N, dx);
u2  = M2 \ rhs;                      

%% --- Verification ---

fprintf('  max|u1 - u2| = %e    <--identical observables\n', max(abs(u1-u2)));

% Kink in W = δD·u = C (constant): [W'](z) should be ~0
W    = (D2 - D1) .* u1;              
dW   = diff(W) / dx;
kink_W = dW(iz) - dW(iz-1);
fprintf('  [W''](z)  = %+.2e  (≈0: W is flat, kink lives in u'')\n', kink_W);

% Kink in u' at z 
du1      = diff(u1) / dx;
kink_u1  = du1(iz) - du1(iz-1);
fprintf('  [u1''](z) = %+.6f  ← C^1 kink', kink_u1);

%% --- Plotting ---
figure('Color','w','Position',[60 80 1500 450]);
ax1 = subplot(1,3,1);
plot(x, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');
ypad = 0.05*(max(u1)-min(u1)); ylim([min(u1)-ypad, max(u1)+ypad]);

ax2 = subplot(1,3,2);
plot(x, D1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, D2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax3 = subplot(1,3,3);
stem(z, b0, '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(z, b0, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',4, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Point sources','FontWeight','normal');
legend({'$b_0^{(1)}$','$b_0^{(2)}$'}, 'Interpreter','latex','Location','northeast');
xlim([0 1]); ylim([0 b0*1.5]);
text(z+0.03, b0*1.08, sprintf('$b_0^{(1)} = b_0^{(2)} = %g$', b0), 'Interpreter','latex','FontSize', 13);

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

exportgraphics(gcf, 'Ito_Neumann_kink.pdf', 'ContentType', 'vector');
