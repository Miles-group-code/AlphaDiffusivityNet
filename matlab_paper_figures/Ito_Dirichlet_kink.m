%% ----Itô Dirichlet Weak (C^0) Doppelgänger ----
clear; clc; close all;

% --- shared figure style (python Okabe-Ito palette) ---
c1 = [0.337 0.706 0.914];   % Model 1 (true)         blue   #56B4E9
c2 = [0.000 0.620 0.451];   % Model 2 (doppelganger) green  #009E73
lw = 2.5;                     % uniform line width

L  = 1;
N  = 4001;
x  = linspace(0, L, N)';
dx = x(2) - x(1);

z   = 0.5;  [~, iz] = min(abs(x - z));
b0_1    = 10;
gamma   = 5;
delta_b0 = 4;
b0_2    = b0_1 + delta_b0;

% D1 = 2 + 0.8*sin(10*pi*x) + 0.5*cos(4*pi*x);
D1 = 2 + 0.8*sin(2*pi*x);

Aw = zeros(N,N);
for i = 2:N-1
    Aw(i,i-1) =  1/dx^2;
    Aw(i,i  ) = -2/dx^2;
    Aw(i,i+1) =  1/dx^2;
end

function u = solve_dir(Aw, D, gamma, b0, iz, dx, N)
    M = Aw * diag(D) - gamma * eye(N);
    M(1,:) = 0; M(1,1) = 1;
    M(N,:) = 0; M(N,N) = 1;
    rhs = zeros(N,1);
    rhs(iz) = -b0 / dx;
    u = M \ rhs;
end

u1 = solve_dir(Aw, D1, gamma, b0_1, iz, dx, N);

%% ---W = delta_b0 * G0(x,z)  (kinks at z)---
G0 = zeros(N,1);
G0(x <= z) = x(x <= z) * (L - z) / L;
G0(x >  z) = z * (L - x(x > z)) / L;
W = delta_b0 * G0;                    % W(0)=0, W(L)=0, W''=-delta_b0*delta(z)

%% deltaD = W / u1
thresh   = 0.02 * max(abs(u1));
mask     = abs(u1) > thresh;
deltaD   = zeros(N,1);
deltaD(mask) = W(mask) ./ u1(mask);

i_first = find(mask, 1, 'first');
i_last  = find(mask, 1, 'last');
deltaD(1:i_first-1)  = deltaD(i_first);
deltaD(i_last+1:end) = deltaD(i_last);

D2 = D1 + deltaD;

u2 = solve_dir(Aw, D2, gamma, b0_2, iz, dx, N);

%% ---Verification ---
max_err = max(abs(u1 - u2));
fprintf('  max|u1-u2| = %.2e  (%.4f%% of peak u1)\n', max_err, max_err/max(u1)*100);

% Kink in W at z: [W'](z) should equal -delta_b0
dW = diff(W) / dx;
kink_W = dW(iz) - dW(iz-1);
fprintf('[W''](z) = %.4f  (expected = %.4f)\n', kink_W, -delta_b0);

% Kink in u1' at z
du1 = diff(u1) / dx;
kink_u1 = du1(iz) - du1(iz-1);
fprintf('  [u1''](z) = %.4f  <- C^1 kink\n', kink_u1);

% Slopes of D1 and D2 near z (shown in the panel-(b) insets): one-sided
% differences, never differencing across z, so a jump shows as a clean step
dD1 = diff(D1) / dx;  dD2 = diff(D2) / dx;   % live on [x(k), x(k+1)]
xm  = (x(1:end-1) + x(2:end)) / 2;
win = z + [-0.06 0.06];               % inset x-window
iL  = find(xm >= win(1) & xm < z);    % segments entirely left of z
iR  = find(xm > z & xm <= win(2));    % segments entirely right of z
fprintf('  [D1''](z) = %.4f   [D2''](z) = %.4f  (%.3f -> %.3f)\n', ...
        dD1(iz) - dD1(iz-1), dD2(iz) - dD2(iz-1), dD2(iz-1), dD2(iz));

%% ---Plot---
fig = figure('Color','w','Position',[60 80 1500 450]);
ax1 = subplot(1,3,1);
plot(x, u1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, u2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$u(x)$','Interpreter','latex'); title('Identical densities','FontWeight','normal');
legend({'$u_1(x)$','$u_2(x)$'}, 'Interpreter','latex','Location','northeast');

ax2 = subplot(1,3,2);
plot(x, D1, '-',  'Color', c1, 'LineWidth', lw); hold on;
plot(x, D2, '--', 'Color', c2, 'LineWidth', lw);
xlabel('$x$','Interpreter','latex'); ylabel('$D(x)$','Interpreter','latex'); title('Distinct diffusivities','FontWeight','normal');
legend({'$D_1(x)$','$D_2(x)$'}, 'Interpreter','latex','Location','southeast');   % panel (b) only: frees top right for the D2' inset

ax3 = subplot(1,3,3);
stem(z, b0_1, '-',  'Color', c1, 'MarkerFaceColor',c1, 'MarkerSize',7, 'LineWidth',lw); hold on;
stem(z, b0_2, '--', 'Color', c2, 'MarkerFaceColor',c2, 'MarkerSize',7, 'LineWidth',lw);
xlabel('$x$','Interpreter','latex'); ylabel('$b_0$','Interpreter','latex'); title('Point sources','FontWeight','normal');
legend({sprintf('$b_0^{(1)} = %g$',b0_1), sprintf('$b_0^{(2)} = %g$',b0_2)}, 'Interpreter','latex','Location','northeast');
xlim([0 1]); ylim([0 b0_2 * 1.5]);

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

%% ---Panel (b) insets: D1' (no jump) and D2' (jump) near z, each tied to a zoom box---
% Panel (b) y-axis starts at 0 to open an empty band under D1 (D1' inset at
% lower left, legend at lower right); the D2' inset sits top right.
ylim(ax2, [0, ax2.YLim(2)]);
xl2 = ax2.XLim; yl2 = ax2.YLim; p = ax2.Position;
d2f = @(xd, yd) [p(1) + (xd - xl2(1))/diff(xl2)*p(3), p(2) + (yd - yl2(1))/diff(yl2)*p(4)];

% shared y-span (data units) so the two insets compare directly
j    = [iL; iR];
span = 1.24 * max(max(dD1(j)) - min(dD1(j)), max(dD2(j)) - min(dD2(j)));
ylim1 = mean([min(dD1(j)) max(dD1(j))]) + [-0.5 0.5]*span;
ylim2 = mean([min(dD2(j)) max(dD2(j))]) + [-0.5 0.5]*span;


bx    = win;                          % zoom boxes span the inset x-window
ipos1 = [p(1)+0.06*p(3), p(2)+0.09*p(4), 0.31*p(3), 0.22*p(4)];   % D1' inset, lower left
ipos2 = [p(1)+0.63*p(3), p(2)+0.67*p(4), 0.31*p(3), 0.22*p(4)];   % D2' inset, top right
by1   = D1(iz) + [-0.30 0.30];
by2   = D2(iz) + [-0.30 0.30];

% Thin dark-grey framing: reads darker than the grid but stays quiet next to the curves
fc = [0.30 0.30 0.30];                % zoom boxes + inset frames
cc = fc;                               % connectors (dotted, same dark grey)
add_zoom_inset(fig, ax2, d2f, fc, cc, bx, by1, ipos1, 'left', ...
    xm, dD1, iL, iR, win, ylim1, z, c1, '$D_1''(x)$');   % D1': connectors from the box's left side
add_zoom_inset(fig, ax2, d2f, fc, cc, bx, by2, ipos2, 'right', ...
    xm, dD2, iL, iR, win, ylim2, z, c2, '$D_2''(x)$');   % D2': connectors from the box's right side

exportgraphics(fig, 'Ito_Dirichlet_kink.pdf', 'ContentType', 'vector');
close(fig);

function add_zoom_inset(fig, ax, d2f, fc, cc, bx, by, ipos, side, xm, dv, iL, iR, win, yli, z, col, lbl)
    % fc: zoom box + inset frame colour; cc: connector colour
    rectangle(ax, 'Position', [bx(1) by(1) diff(bx) diff(by)], 'EdgeColor', fc, 'LineWidth', 0.6);
    % inset: one-sided slopes on each side of z, dotted line at z
    ins = axes(fig, 'Position', ipos);  hold(ins, 'on');
    plot(ins, [z z], yli, ':', 'Color', [0.65 0.65 0.65], 'LineWidth', 0.75);
    plot(ins, xm(iL), dv(iL), '-', 'Color', col, 'LineWidth', 2);
    plot(ins, xm(iR), dv(iR), '-', 'Color', col, 'LineWidth', 2);
    xlim(ins, win); ylim(ins, yli);
    box(ins, 'on'); grid(ins, 'off');
    set(ins, 'XTick', [], 'YTick', [], 'LineWidth', 0.6, 'Color', 'w', 'XColor', fc, 'YColor', fc);   % opaque white: hides panel grid
    title(ins, lbl, 'Interpreter','latex', 'FontSize', 15, 'FontWeight','normal', 'Color', 'k', ...
          'Units','normalized', 'Position', [0.5 1.01], 'VerticalAlignment','bottom');
    % connectors from the near side of the box to the facing side of the inset
    switch side
        case 'left'    % inset left of the box
            a = d2f(bx(1), by(2));  b = d2f(bx(1), by(1));  xi = ipos(1) + ipos(3);
        case 'right'   % inset right of the box
            a = d2f(bx(2), by(2));  b = d2f(bx(2), by(1));  xi = ipos(1);
    end
    annotation(fig, 'line', [a(1) xi], [a(2) ipos(2)+ipos(4)], 'Color', cc, 'LineWidth', 1.0, 'LineStyle', ':');
    annotation(fig, 'line', [b(1) xi], [b(2) ipos(2)],         'Color', cc, 'LineWidth', 1.0, 'LineStyle', ':');
end
