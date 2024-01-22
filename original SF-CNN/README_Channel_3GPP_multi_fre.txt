clear
Nc=64;
L=3; % number of path
Nt=32;
Nr=16;




samplingrate=1e8;
num_fre=2;
num_sta=50;
num_ffading=200;

################# chương trình lấy dữ liệu từ tham số kênh truyền gen ngẫu nhiên để biến đổi thành hàm truyền thu được đầu thu 
load ch_sta_mtx

ChannelData_fre=zeros(Nt,Nr,num_fre,num_sta*num_ffading);

################## Tham số CDL lấy từ tham số của channel stats, dữ liệu lấy từ code gen kênh truyền
sta_mtx=channel_statistic(:,:,m); ###############lấy từ code tham số kênh truyền
LOSangle=60;           ########## Không quan trọng
philosAoA=LOSangle;    ########## Không quan trọng
philosAoD=LOSangle;    ########## Không quan trọng
thetalosZoA=LOSangle;  ########## Không quan trọng
thetalosZoD=LOSangle;  ########## Không quan trọng

####### Tạo hàm truyền theo 
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
                         Seed: 2000
      NormalizeChannelOutputs: true
             ChannelFiltering: false
               NumTimeSamples: 30720
               OutputDataType: 'double'
    TransmitAndReceiveSwapped: false


###### Gen hàm truyền
for n=1:num_ffading
    cdl.Seed = (l-1)*num_ffading+n;
    [pathgains,sampletimes]=step(cdl);
    pathgains=sqrt(Nr)*pathgains;
    for nt=1:Nt
        for nr=1:Nr
            pathpower=pathgains(:,:,nt,nr);
            h=zeros(1,1024);
            I=floor(cdl.PathDelays*samplingrate)+1;
            I_uniq=unique(I);
            Power_sum=[];
            for i=1:length(I_uniq)
                Power_sum(i)=sum(pathpower(I==I_uniq(i)));
            end
            h(I_uniq)=Power_sum;
            h_ntnr=h(1:Nc);
            h_fre=fft(h_ntnr);            
            fre=32;
            pathgains_fre(:,nt,nr)=h_fre(fre:fre+num_fre-1);   %  frequency domain channel matrix
        end
    end
    for j=1:num_fre
        pathgains_fre2(:,:,j)=pathgains_fre(j,:,:);
    end
    ChannelData_fre(:,:,:,(m-1)*num_ffading+n)=pathgains_fre2;
    release(cdl);
end
end
