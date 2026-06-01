%% --- Fickian Model with Robin Boundary Figure 6 ---
clear; clc; close all;

% D1(x) and its derivative
D1     = @(x) 1.5 + sin(4*x);
D1p    = @(x) 4*cos(4*x);

b1 = @(x) 1.0 * (1./(0.1*sqrt(2*pi))) .* exp(-0.5*((x - 0.5)./0.1).^2);

N = 1e4;
xmesh = linspace(0,1,N);

% Fickian ODE: D1*u'' + D1p*u' + b1 = 0
odefun1 = @(x,y) [ ...
    y(2); ...
    ( -b1(x) - D1p(x).*y(2) ) ./ D1(x) ...
];

%% --- Robin Boundary Conditions ---
% Form: alpha * u + beta * u' = gamma
% Left boundary (x = 0):  1 * u(0) - 0.5 * u'(0) = 0.2
% Right boundary (x = 1): 1 * u(1) + 0.8 * u'(1) = 2.0
alpha_a = 1.0; beta_a = -0.5; gamma_a = 0.2;
alpha_b = 1.0; beta_b = 0.8;  gamma_b = 2.0;

bcfun = @(ya,yb) [ ...
    alpha_a * ya(1) + beta_a * ya(2) - gamma_a; ... % Robin BC at x = 0
    alpha_b * yb(1) + beta_b * yb(2) - gamma_b  ... % Robin BC at x = 1
];


sol1    = bvp4c(odefun1, bcfun, bvpinit(xmesh, @(x)[1.5*x + 0.2; 1.5]));
y1      = deval(sol1, xmesh);
u1      = y1(1,:)';                   
u1p     = y1(2,:)';                   

% Integrate b1 to find the flux profile
b1_vals = b1(xmesh)';
% Phi = cumtrapz(xmesh, b1_vals); 

% D2(x) = D1(x) + C_flux / u'(x)
C_flux = 0.5;
D2 = D1(xmesh)' + (C_flux ./ u1p);

h = xmesh(2) - xmesh(1);
D2p  = gradient(D2, h);

D2i   = @(x) interp1(xmesh, D2,   x, 'pchip', 'extrap');
D2pi  = @(x) interp1(xmesh, D2p,  x, 'pchip', 'extrap');
b2 = b1;

odefun2 = @(x,y) [ ...
    y(2); ...
    ( -b2(x) - D2pi(x).*y(2) ) ./ D2i(x) ...
];

sol2 = bvp4c(odefun2, bcfun, bvpinit(xmesh, @(x) deval(sol1,x)));
y2  = deval(sol2, xmesh);
u2  = y2(1,:)';

%% --- Plot ---
diff_norm = norm(u1 - u2);
fprintf('||u1 - u2||_2 = %.3e\n', diff_norm);
flux1 = D1(xmesh)'.*u1p;
flux2 = D2.*u1p;
C_test = flux2 - flux1;
fprintf('std(flux2-flux1) = %.3e\n', std(C_test));
fprintf('min(D2) = %.4e\n', min(D2));
fprintf('max(D2) = %.4e\n', max(D2));
fprintf('min(abs(u1p)) = %.4e\n', min(abs(u1p)));

figure('Color', 'w', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1);
plot(xmesh,u1,'b-','LineWidth',2,'DisplayName','u_1(x)'); hold on;
plot(xmesh,u2,'r--','LineWidth',2,'DisplayName','u_2(x)');
box off; legend('Location','northwest'); title('Identical Solutions u_1 \equiv u_2');
xlabel('x'); ylabel('u(x)'); grid on;

subplot(1,3,2);
plot(xmesh, D1(xmesh),'b-','LineWidth',2,'DisplayName','D_1(x)'); hold on;
plot(xmesh, D2,'r--','LineWidth',2,'DisplayName','D_2(x)');
box off; legend('Location','best'); title('Distinct Diffusion Coefficients'); 
xlabel('x'); ylabel('D(x)'); grid on;

subplot(1,3,3);
plot(xmesh, b1(xmesh), 'b-', 'LineWidth', 2, 'DisplayName', 'b_1(x)');hold on;
plot(xmesh, b2(xmesh), 'r--', 'LineWidth', 2, 'DisplayName', 'b_2(x)');
legend('Location', 'best'); title('Identical Source Terms');
xlabel('x'); ylabel('Source Magnitude'); grid on;