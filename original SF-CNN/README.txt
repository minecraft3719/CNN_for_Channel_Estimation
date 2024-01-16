original code

1. Tạo mô hình kênh
*Thông số hệ thống đã biết trước


Nc=64;
L=3;         Số đa đường
Nt=32;       Antenna phát
Nr=16;       Antenna thu      
num_para=6;  ######### Chưa rõ thông số ###########
num_sta=50;  ######### Chưa rõ thông số ###########

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

