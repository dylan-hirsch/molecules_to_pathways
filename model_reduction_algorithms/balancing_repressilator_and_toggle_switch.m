r = 5; % size of reduced basis

K1 = .25;
n1 = 4;
K2 = .25;
n2 = 2;

%% Get relevant roots

x_represillator = fzero(@(x) represillator_root(x,K1,n1), 1);
x_toggle = fzero(@(x) toggle_root(x,K2,n2), 1);

%% Generate system matrices

a = mm_derivative(x_represillator, K1, n1);
b = mm_derivative(x_toggle, K2, n2);
c = mm(x_represillator, K1, n1);

A11 = [-1 0 a; a -1 0; 0 a -1];
[U,E] = eigs(A11);
for i = 1:3
    if real(E(i,i)) > 0
        E(i,i) = -E(i,i);
    end
end
A11 = U * E * pinv(U);
A22 = [-1 b; b -1];

A = [A11, zeros(3,2); zeros(2,3), A22];
B = [0; 0; 0; c; c];
C = [0 0 0 0 1];
D = 0;

%% Do balanced truncation

R = lyapchol(A,B)';
L = lyapchol(A',C')';

[U,S,V] = svd(L' * R);
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

Tr = R * Vr * pinv(sqrt(Sr));
Trinv = pinv(sqrt(Sr)) * Ur' * L';

%% Get new system matrices

Arb = Trinv * A * Tr;
Brb = Trinv * B;
Crb = C * Tr;
Drb = D;

%% Functions

function [y] = mm(x, K, n)
    y = 1 / (1 + (x/K)^n);
end

function [dy] = mm_derivative(x, K, n)
    dy = - n * x^(n-1) / K^n / (1 + (x/K)^n)^2;
end

function [y] = represillator_root(x, K, n)
    y = x^(n+1) + K^n * x - K^n;
end

function [y] = toggle_root(x, K, n)
    y = K^n * (1 + (x / K)^n)^n * (1 - x) - x;
end

