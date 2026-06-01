%% ----Itô Dirichlet Weak (C^0) Doppelgänger — Figure 4 ----
clear; clc; close all;


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
fprintf('  [u1''](z) = %.4f  <- C^1 kink', kink_u1);

%% ---Plot---
figure('Color','w','Position',[60 80 1400 440]);
subplot(1,3,1);
plot(x, u1, 'b-',  'LineWidth', 2, 'DisplayName', 'u_1(x)'); hold on;
plot(x, u2, 'r--', 'LineWidth', 2, 'DisplayName', 'u_2(x)');
xline(z, 'k:', 'LineWidth', 1.5,'DisplayName', 'x=0.5');
xlabel('x'); ylabel('u');
title({'Comparison of u_1(x) and u_2(x)'});
legend('Location','best'); grid on;

subplot(1,3,2);
plot(x, D1, 'b-',  'LineWidth', 2,   'DisplayName', 'D_1(x)'); hold on;
plot(x, D2, 'r--', 'LineWidth', 2,   'DisplayName', 'D_2(x)');
xline(z, 'k:', 'LineWidth', 1.5,'DisplayName', 'x=0.5');
xlabel('x'); ylabel('D');
title({'Different D_1(x) and D_2(x)'})
legend('Location','best'); grid on;

subplot(1,3,3);
stem(z, b0_1, 'b-', 'MarkerFaceColor','b', 'MarkerSize', 8, ...
     'LineWidth', 2, 'DisplayName', sprintf('b_0^{(1)} = %g', b0_1));
hold on;
stem(z, b0_2, 'r--', 'MarkerFaceColor','r', 'MarkerSize', 8, ...
     'LineWidth', 2, 'DisplayName', sprintf('b_0^{(2)} = %g', b0_2));
xlabel('x'); ylabel('Magnitude');
title('Point Sources b(x) with Different Magnitude');
xlim([0 1]); ylim([0 b0_2 * 1.5]);
legend('Location','best'); grid on;