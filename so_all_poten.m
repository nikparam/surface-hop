
clear all

dt = 0.01;
TSteps = 500;
N = 2;

k0 = 5;
x0 = -10;

k = 0.81;
mass = 1;

omega = sqrt(k/mass);
sigma = sqrt(2/mass/omega);
sigma_inv = 1/sigma;
Norm = sqrt(1/sqrt(2*pi*sigma));

pswitch = -4;

A = [0.01,0.1,0.0006];
B = [1.6,0.28,0.1];
C = [0.005,0.015,0.9];
D = [1.,0.06];
E_0 = 0.05;
shift = 3;

Npoints = 1024;
Lim = 15;
Mapping = 2*Lim;
delta = Mapping/Npoints;
f0 = 2*pi/Mapping;
x_space = linspace(-Lim,Lim,Npoints);
k_space = linspace(-Npoints*f0*0.5,(0.5*Npoints-1)*f0,Npoints);

mean_x = zeros(N,TSteps);
mean_k = zeros(N,TSteps);
d_population = zeros(N,TSteps);
ad_population = zeros(N,TSteps);

psi0 = zeros(N,Npoints);
initState = 1;
temp = Norm*exp(-0.25*sigma_inv*(x_space-x0).^2).*exp(1i*k0*x_space);
psi0(initState,:) = reshape(temp,1,Npoints);
psi0(initState,:) = psi0(initState,:)/sqrt(psi0(initState,:)*psi0(initState,:)');

%  figure(1);
%  plot(x_space,abs(psi0(initState,:)),'k','LineWidth',3);

kinetic = k_space.^2/2/mass;
T_prop = exp(-1i*kinetic.*dt);

VV = zeros(2,2);
for i=1:Npoints
    x = x_space(i);
    if (pswitch == 1) || (pswitch == -1)
        if x >= 0
            VV(1,1) = A(1)*(1-exp(-B(1)*x));
        else
            VV(1,1) = -A(1)*(1-exp(B(1)*x));
        end
        VV(2,2) = -VV(1,1);
        VV(1,2) = C(1)*exp(-D(1)*x*x);
    end   
    if (pswitch == 2) || (pswitch == -2)
        VV(1,1) = 0;
        VV(2,2) = -A(2)*exp(-B(2)*x*x)+E_0;
        VV(1,2) = C(2)*exp(-D(2)*x*x);
        VV(2,1) = VV(1,2);
    end
    if (pswitch == 3) || (pswitch == -3)
        VV(1,1) = -A(3);
        VV(2,2) = A(3);
        if x < 0
            VV(1,2) = B(3)*exp(C(3)*x);
        else
            VV(1,2) = B(3)*exp(-C(3)*x);
        end
    end
    if (pswitch == 4) || (pswitch == -4)
        VV(1,1) = 0.5*k*x^2;
        VV(2,2) = 0.5*k*(x-shift)^2-omega;
        VV(1,2) = 0.5*omega;
        VV(2,1) = VV(1,2);
    end
    
    [eigVec,eigVal] = eig(VV);
    [eigVal,index] = sort(diag(eigVal));    eigVec = eigVec(:,index);
    V_prop = eigVec*diag(exp(-1i*dt*eigVal))*eigVec';
    
    eigVec_1(:,i) = conj(eigVec(:,1));   eigVec_2(:,i) = conj(eigVec(:,2));
    V1(:,i) = VV(:,1);    V2(:,i) = VV(:,2);
    V_prop_1(:,i) = V_prop(:,1);   V_prop_2(:,i) = V_prop(:,2);
    ev1(i) = eigVal(1);    ev2(i) = eigVal(2);
    
end

 psi = psi0;
 psi00 = psi0;
 for t=1:TSteps,
     E(t) = 0;
     psi = V_prop_1.*[psi(1,:);psi(1,:)]+V_prop_2.*[psi(2,:);psi(2,:)];
     for n=1:2,
        temp = psi(n,:);
        temp = fftshift(fft(temp));
        mean_k(n,t) = real(sum(sum(conj(temp).*k_space.*temp)))./sum(sum(conj(temp).*temp));
        temp = temp.*T_prop;
        E_kin(t,n) = real(sum(sum(conj(temp).*kinetic.*temp)))./sum(sum(conj(temp).*temp));
        temp = ifft(fftshift(temp));
        psi(n,:) = temp;
     end
     ket = V1.*[psi(1,:);psi(1,:)]+V2.*[psi(2,:);psi(2,:)];
     E_pot(t) = real(sum(sum(conj(psi).*ket)))./sum(sum(conj(psi).*psi));
     d_population(:,t) = sum(conj(psi).*psi,2);
      
     psi_ad = eigVec_1.*[psi(1,:);psi(1,:)]+eigVec_2.*[psi(2,:);psi(2,:)];
     ad_population(:,t) = sum(conj(psi_ad).*psi_ad,2);
      
     E(t) = E_pot(t)+E_kin(t,1)*d_population(1,t)+E_kin(t,2)*d_population(2,t);
     Norm1(t) = real(sum(sum(conj(psi).*psi0)));
     autocorr(t) = real(sum(sum(conj(psi).*psi00)));
     psi0 = psi;
     for n=1:2
         mean_x(n,t) = real(sum(sum(conj(psi(n,:)).*x_space.*psi(n,:))))./sum(sum(conj(psi(n,:)).*psi(n,:)));
     end
%      linkdata on;
%      figure(1)
%      plot(x_space,ev1,'black',x_space,ev2,'black',x_space,500*abs(psi(1,:)),'r',x_space,500*abs(psi(2,:)),'b','LineWidth',3)
 end

   time = linspace(0,dt*TSteps,TSteps);
   figure(2);
%    plot(time,d_population(1,:),'r',time,d_population(2,:),'b');
plot(time,E) 
