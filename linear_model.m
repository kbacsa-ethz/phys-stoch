clear all; close all;
m1 = 1;
m2 = 1;
k1 = 2.9;
k2 = 0.6;

M = diag([m1,m2]);
K = [k1+k2 -k2;
     -k2 k2];
C = diag([0,0]);
A = [zeros(2,2), eye(2);
    -M\K, -M\C];

dt = 0.05;
Ad = expm(A*dt);

t_len =200;
y = zeros(4,t_len);

y0 = zeros(4,1);
y0(1) = 2;
y0(2) = 1;
y0(3) = 1;
y0(4) = -2;
for i = 1:t_len
    
    y(:,i) = Ad*y0;
    
    y0 = y(:,i);
end

figure()
plot(y')

T = 0.5*m1*y(3,:).^2 + 0.5*m2*y(4,:).^2;
V = 0.5 * k1* y(1,:).^2 +  0.5 * k2* (y(2,:) - y(1,:)).^2 ;
figure()
plot(T,'-b'); hold on
plot(V,'-k')
plot(T + V,'-g')
plot(T - V,'-r')

legend("kinetic energy", "potential energy", "total energy","T - V")
