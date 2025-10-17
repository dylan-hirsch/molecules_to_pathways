r = 3; % size of reduced basis

A = [-1 0 0 ; 1 -1 0; 0 1 -1];
B = [1 .1 0 0; 0 0 .1 0; 0 0 0 .1];
C = [0 0 1];
D = 0;

R = lyapchol(A,B)';
L = lyapchol(A',C')';

[U,S,V] = svd(L' * R);
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

Tr = R * Vr * pinv(sqrt(Sr));
Trinv = pinv(sqrt(Sr)) * Ur' * L';

Arb = Trinv * A * Tr;
Brb = Trinv * B;
Crb = C * Tr;
Drb = D;