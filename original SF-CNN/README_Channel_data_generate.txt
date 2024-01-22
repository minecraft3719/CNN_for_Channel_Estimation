original code

1. Tạo mô hình kênh
##########Thông số hệ thống đã biết trước
Nc=64;
L=3;         Số đa đường
Nt=32;       Antenna phát
Nr=16;       Antenna thu      
num_para=6;  Số thông số muốn lưu lại bao gồm 6 thông số Path delay, Power, AoA, AoD, ZoA, ZoD
num_sta=50;  Số lượng mẫu kênh muốn lưu lại để training
channel_statistic=zeros(num_para,L,num_sta); ########### tập dữ liệu mô hình kênh truyền


góc quét 
LOSangle=60;
philosAoA=LOSangle;
philosAoD=LOSangle;
thetalosZoA=LOSangle;
thetalosZoD=LOSangle;

cdl = nrCDLChannel;
để tạo ra channel, code sử dụng mô hình cdl channel (Clustered Delay Line) - Chi tiết: https://www.mathworks.com/help/5g/ref/nrcdlchannel-system-object.html
Thông số nrCDLChannel
                 DelayProfile: 'Custom'
                   PathDelays: [0 1.0717e-07 5.3923e-07]
             AveragePathGains: [-2.9143 -3.5454 -13.2990]
                    AnglesAoD: [59.6457 48.5160 98.3693]
                    AnglesAoA: [53.9585 28.4774 280.3990]
                    AnglesZoD: [54.5326 54.2661 57.8341]
                    AnglesZoA: [53.9749 58.3721 121.4423]
                HasLOSCluster: false
                 AngleSpreads: [5 11 3 3]
                  RayCoupling: 'Random'
                          XPR: 10
                InitialPhases: 'Random'
             CarrierFrequency: 2.8000e+10
          MaximumDopplerShift: 0
          UTDirectionOfTravel: [2×1 double]
                   SampleRate: 30720000
         TransmitAntennaArray: [1×1 struct]
     TransmitArrayOrientation: [3×1 double]
          ReceiveAntennaArray: [1×1 struct]
      ReceiveArrayOrientation: [3×1 double]
           NormalizePathGains: true
                SampleDensity: 64
                  InitialTime: 0
         NumStrongestClusters: 0
                 RandomStream: 'mt19937ar with seed'
                         Seed: 73
      NormalizeChannelOutputs: true
             ChannelFiltering: false
               NumTimeSamples: 30720
               OutputDataType: 'double'
    TransmitAndReceiveSwapped: false

Tính các thành phần thông số trong hàm chuyền
######Tính Path delay
    gDSmean=-0.24*log10(1+ fc) - 6.83;   %UMi - Street Canyon  NLOS - Delay spread
    lgDSstanvar=0.16*log10(1+ fc) + 0.28;
    DS=10^(normrnd(lgDSmean,lgDSstanvar));
    r_tau=2.1;
    tau=-r_tau*DS*log(rand(1,L));
    cdl.PathDelays=sort(tau-min(tau));
    save_PD=cdl.PathDelays;   

######Tính công suất kênh truyền - tương ứng với độ lợi trung bình tổng các đường
    Power=exp(-(r_tau-1)/(r_tau*DS)*cdl.PathDelays).*10.^(-normrnd(0,3,1,L)/10);
    P=Power/sum(Power);
    cdl.AveragePathGains=10*log10(P);  % average path gains
    save_P=cdl.AveragePathGains;

######Tính góc tới AoA
lgASAmean=-0.08*log10(1+ fc) + 1.81;   %UMi - Street Canyon  NLOS - Angle spread of Arrival
lgASAstanvar=0.05*log10(1+ fc) + 0.3;
ASA=10^(normrnd(lgASAmean,lgASAstanvar));  % ASA tính ngẫu nhiên theo ASA
Cphi=0.779; % N=4
Phi=2*ASA/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoA=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASA/7,1,L)+philosAoA;  % AoA
save_AoA=cdl.AnglesAoA;

######Tính góc đi AoD - Cách gen ra giưx liệu giống với AoA nhưng sẽ dùng biến độc lập
lgASDmean=-0.23*log10(1+ fc) + 1.53;   %UMi - Street Canyon  NLOS - Angle spread of Departure
lgASDstanvar=0.11*log10(1+ fc) + 0.33;
ASD=10^(normrnd(lgASDmean,lgASDstanvar));  % ASD
Cphi=0.779; % N=4
Phi=2*ASD/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoD=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASD/7,1,L)+philosAoD;  % AoD
save_AoD=cdl.AnglesAoD;

######Tính góc ZoA - bổ sung và không có trong bài báo
lgZSAmean=-0.04*log10(1+ fc) + 0.92;   %UMi - Street Canyon  NLOS
lgZSAstanvar=-0.07*log10(1+ fc) + 0.41;
ZSA=10^(normrnd(lgZSAmean,lgZSAstanvar));  % ZSA
Ctheta=0.889; % N=8
Theta=-ZSA*log(P/max(P))/Ctheta;
cdl.AnglesZoA=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSA/7,1,L)+thetalosZoA;  % ZoA
save_ZoA=cdl.AnglesZoA;

######Tính góc ZoD, bổ sung, không có trong bài báo và thêm một số tham số độc lập 
d2D=50;
hUT=0;
hBS=10;
lgZSDmean=max(-0.5, -3.1*(d2D/1000)+ 0.01*max(hUT-hBS,0) +0.2);   %UMi - Street Canyon  NLOS
lgZSDstanvar=0.35;
ZSD=10^(normrnd(lgZSDmean,lgZSDstanvar));  % ZSD
Ctheta=0.889; % N=8
Theta=-ZSD*log(P/max(P))/Ctheta;
cdl.AnglesZoD=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSD/7,1,L)-10^(-1.5*log10(max(10, d2D))+3.3)+thetalosZoD;  % ZoD
save_ZoD=cdl.AnglesZoD;

###### Save cấu hình 
Dữ liệu được lưu lại để chuyển sang training phase
sta_mtx=[save_PD;save_P;save_AoA;save_AoD;save_ZoA;save_ZoD];
channel_statistic(:,:,m)=sta_mtx; ############ Để có nhiều tập dữ liệu hàm truyền training khác nhau, thông số mảng được lưu lại thành nhiều mẫu

save ch_sta_mtx channel_statistic
