%% ---Itô With Neumann Boundary (Weak Doppelgänger) — Figure 3 ---

clear; clc; close all;

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
figure('Color','w','Position',[60 80 1300 420]);
subplot(1,3,1);
plot(x, u1, 'b-',  'LineWidth', 2, 'DisplayName', 'u_1(x) (True)'); hold on;
plot(x, u2, 'r--', 'LineWidth', 2, 'DisplayName', 'u_2(x) (Phantom)');
xline(z, 'k:', 'LineWidth', 1.5, 'DisplayName', 'x=0.5');
xlabel('x'); ylabel('u');
title('Comparison of u_1(x) and u_2(x)');
legend('Location','best'); grid on;
ypad = 0.05*(max(u1)-min(u1));
ylim([min(u1)-ypad, max(u1)+ypad]);

subplot(1,3,2);
plot(x, D1, 'b-',  'LineWidth', 2, 'DisplayName', 'D_1(x)'); hold on;
plot(x, D2, 'r--', 'LineWidth', 2, 'DisplayName', 'D_2(x)');
xline(z, 'k:', 'LineWidth', 1.5,'DisplayName', 'x=0.5');
xlabel('x'); ylabel('D');
title('Different D_1(x) and D_2(x)');
legend('Location','best'); grid on;

subplot(1,3,3);
stem(z, b0, 'b-', 'MarkerFaceColor','b', 'MarkerSize',8, ...
     'LineWidth', 2, 'DisplayName', 'b_1 (True)'); hold on;
stem(z, b0, 'r--', 'MarkerFaceColor','r', 'MarkerSize',4, ...
     'LineWidth', 2, 'DisplayName', 'b_2 (Phantom)');
xlabel('x'); ylabel('Magnitude');
title('Point Sources b(x) with Same Magnitude');
xlim([0 1]); ylim([0 b0*1.5]);
legend('Location','best'); grid on;
text(z+0.03, b0*1.08, sprintf('b_1 = b_2 = %g', b0), ...
     'FontSize', 9, 'Color','k');
