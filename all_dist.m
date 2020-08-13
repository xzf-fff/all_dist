clear
%% 读取图像
img=imread("ship14.tif");
[MM,NN]=size(img);
img=im2uint8(img);
% img=img(1000:2000,1000:2000);
%% 直方图
[Freq,XX]=hist(double(img(:)),250);
binWidth=XX(2)-XX(1);
bar(XX,Freq/binWidth/sum(Freq));
%% 预处理
img=double(img);
index=find(img<=1);
img(index)=2;
%% 计算统计量
m1=mean(img,'all');
m2=mean(img.*img,'all');
m4=mean(img.^4,'all');
mlog2=mean(log(img).*log(img),'all');
logm2=log(mean(img.*img,'all'));
mlog=mean(log(img),'all');
mlogmz=mean(log(img.*img),'all');
zlogmz=mean(img.*img.*log(img.*img),'all');
X=zlogmz/m2-mlogmz;
U=mlogmz-logm2;
M=0.15*X+0.85*U;
y1=mlogmz;
y2=mean((log(img.*img)-y1).^2,'all');
y3=mean((log(img.*img)-y1).^3,'all');
%% 参数估计
%高斯分布
u1=mean(img,'all');
sigma21=mean(img.*img,'all')-u1*u1;
%对数正态
u_log=mean(log(img),'all');
sigma2_log=mean(log(img).^2,'all')-(mean(log(img),'all'))^2;
%韦伯
p=(6/(pi*pi)*(MM*NN)/(MM*NN-1)*(mlog2-mlog^2))^(-0.5);
q=exp(mlog+0.5772*p^(-1));
%k
v_V=(m4/(2*m2^2)-1)^(-1);
v_X=1/(X-1);
c_V=2*sqrt(v_V/m2);
c_X=2*sqrt(v_X/m2);
func_handle_k_U=@(x) psi(x)-log(x)-0.5772-U;
func_handle_k_M=@(x) 0.15*(1+1/x)+0.85*(psi(x)-log(x)-0.5772)-M;
func_handle_k_mellin=@(x) func_k_mellin(x,y1,y2,y3);
v_U=fsolve(func_handle_k_U,1);
v_M=fsolve(func_handle_k_M,1);
c_U=2*sqrt(v_U/m2);
c_M=2*sqrt(v_M/m2);
k_mellin_params=fsolve(func_handle_k_mellin,[1,1,0.02]);
%G0
func_handle_g0_mellin=@(x) func_g0_mellin(x,y1,y2,y3);
g0_mellin_params=fsolve(func_handle_g0_mellin,[-2,4,100]);
a_g0=g0_mellin_params(1);
n_g0=g0_mellin_params(2);
r_g0=g0_mellin_params(3);
%% 绘制估计的概率密度
x=0.1:255;
f_norm=1/(sqrt(2*pi*sigma21))*exp(-0.5*(x-u1).^2./(2*sigma21));
f_lognorm=1./(x.*sqrt(sigma2_log*2*pi)).*exp(-(log(x)-u_log).^2/(2*sigma2_log));
f_webu=p/q*(x./q).^(p-1).*exp(-(x./q).^p);
f_k_V=2*c_V/gamma(v_V).*(c_V*x./2).^v_V.*besselk(v_V-1,c_V*x);
f_k_X=2*c_X/gamma(v_X).*(c_X*x./2).^v_X.*besselk(v_X-1,c_X*x);
f_k_U=2*c_U/gamma(v_U).*(c_U*x./2).^v_U.*besselk(v_U-1,c_U*x);
f_k_M=2*c_M/gamma(v_M).*(c_M*x./2).^v_M.*besselk(v_M-1,c_M*x);
f_k_mellin=4*k_mellin_params(3)*k_mellin_params(2).*x./(gamma(k_mellin_params(1))*gamma(k_mellin_params(2))).*(k_mellin_params(3)*k_mellin_params(2).*x.^2).^((k_mellin_params(1)+k_mellin_params(2))/2-1).*besselk(k_mellin_params(1)-k_mellin_params(2),2.*x.*sqrt(k_mellin_params(3)*k_mellin_params(2)));
f_g0_mellin=2*n_g0^n_g0*gamma(n_g0-a_g0)*r_g0^(-a_g0).*x.^(2*n_g0-1)./(gamma(n_g0)*gamma(-a_g0)*(r_g0+n_g0*x.^2).^(n_g0-a_g0));
% hold on;
% plot(x,f_k_V,'LineWidth',2);
% hold on;
% plot(x,f_k_X,'LineWidth',2);
% hold on;
% plot(x,f_k_U,'LineWidth',2);
% hold on;
% plot(x,f_k_M,'LineWidth',2);
% hold on;
% plot(x,f_k_mellin,'LineWidth',2);
% hold on;
% plot(x,f_g0_mellin,'LineWidth',2);
% hold on;
% plot(x,f_norm,'LineWidth',2);
% hold on;
% plot(x,f_lognorm,'LineWidth',2);
% hold on;
% plot(x,f_webu,'LineWidth',2);
% 
% legend("直方图",'K分布（V估计器）','K分布（X估计器）','K分布（U估计器）','K分布（M估计器）','K分布（mellin）','g0分布','正态分布','对数正态分布','韦伯分布');
subplot(3,3,1)
plot(x,f_k_V,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('K分布(V估计器）');
subplot(3,3,2)
plot(x,f_k_X,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('K分布(X估计器）');
subplot(3,3,3)
plot(x,f_k_U,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('K分布(U估计器）');
subplot(3,3,4)
plot(x,f_k_M,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('K分布(M估计器）');
subplot(3,3,5)
plot(x,f_k_mellin,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('K分布(mellin）');
subplot(3,3,6)
plot(x,f_g0_mellin,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('g0分布');
subplot(3,3,7)
plot(x,f_norm,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('正态分布');
subplot(3,3,8)
plot(x,f_lognorm,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('对数正态分布');
subplot(3,3,9)
plot(x,f_webu,'LineWidth',2);
hold on
bar(XX,Freq/binWidth/sum(Freq));
title('韦伯分布');


function y = func_k_mellin(x,y1,y2,y3)
y(1)=(psi(x(1))+psi(x(2))-log(x(3)*x(2))-y1);
y(2)=(psi(1,x(1))+psi(1,x(2))-y2);
y(3)=(psi(2,x(1))+psi(2,x(2))-y3);
end
function y = func_g0_mellin(x,y1,y2,y3)
y(1)=-psi(-x(1))+psi(x(2))+log(x(3)/x(2))-y1;
y(2)=psi(1,-x(1))+psi(1,x(2))-y2;
y(3)=psi(2,-x(1))+psi(2,x(2))-y3;
end

