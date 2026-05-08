function Beam = MP_Bartlett_sim_kraken(ReplicaFile, RwBf, gridDepth, gridRange, rd, rr)
% Bartlett匹配处理，常规处理器，线性处理器
% RwBf: Nfreq*Nr*Nframe   实际接收信号带宽内频谱，频率个数*阵元个数*快拍数
% ReplicaFile: %    带宽内的拷贝场文件名
% Beam: Nsz*Nsr     匹配场模糊表面
% method:           相干或非相干求和
% 在深海100米浅海波导下，计算Nsz*Nsr*Nrd*Nfreq=100*101*200*101的拷贝场
% 与Nrd*Nfreq=200*101的测量场的匹配场处理需要8s,
%% 读取拷贝场文件头部信息
[fid,errmsg] = fopen(ReplicaFile, 'r');
% 2*4 byte 频率个数Nfreq, 匹配场深度网格数Nsz，匹配场距离网格数Nsr, 阵元个数Nr
Nfreq = fread(fid, 1, 'uint32');   
Ndepth = fread(fid, 1, 'uint32');

% (Nfreq+Nsz+Nsr+Nrz+Nsr*Nrr)*8byte  ,其中Nrz = Nrr = Nr = 阵元个数
freqVec = fread(fid, Nfreq,'single');   
depth = fread(fid, Ndepth,'single');    
Nmodes = fread(fid, Nfreq, 'single');      % 每个频率对应的模态个数

origin = ftell(fid);          % 记录当前读取位置
% Beam = zeros(Nsz, Nsr);        % 匹配场模糊表面
Nsz = length(gridDepth);
Nsr = length(gridRange);
Nrd = length(rd);
Beam = zeros(Nsz, Nsr);
%% 求测量场互谱密度矩阵CSDM的估计（在快拍维度上求测量场互谱密度矩阵的平均）
% RwBf: Nfreq*Nr*Nframe
[~, ~, Nframe] = size(RwBf);
if Nframe == 1
    RwBf = normalize(RwBf.', 'norm');      % 对阵列归一化
    Rf = reshape(RwBf, Nrd, 1, Nfreq).*reshape(conj(RwBf), 1, Nrd, Nfreq); %  互谱密度矩阵 Rf: Nr*Nr*Nfreq
else
    A = sqrt(sum(RwBf.*conj(RwBf), 2));        % 求归一化系数
    RwBf = RwBf./A;                         % 对阵列归一化
    Rf = zeros(Nrd, Nrd, Nfreq);
    for ifreq = 1:Nfreq
        Pf = squeeze(RwBf(ifreq, :, :));
        Rf(:, :, ifreq) = Pf*Pf'/Nframe;      % 对快拍平均后的互谱密度矩阵 Rf: Nr*Nr*Nfreq
    end
end



%% 读取拷贝场并作匹配场处理
% 读取频谱范围和声源阵列坐标所对应的内存上的数据
% 数据存储格式为: % (Nsz*Nsr+Nsz*Nsr)*Nrd*Nfreq*4 byte
% offset=(ifreq-1)*Nrd*(Nsz*Nsr+Nsz*Nsr)*4;
% fseek(fid, offset, 'cof');
% tic;

gridDepthIdx = nan*zeros(length(gridDepth),1);
for i = 1:length(gridDepth)
    gridDepthIdx(i) = find( depth == gridDepth(i));
end
rdIdx = nan*zeros(length(rd),1);
for i = 1:Nrd
    rdIdx(i) = find( depth == rd(i) );
end
% 相干积分
rhozs=1;
Q=1i*exp(1i*pi/4)/rhozs*sqrt(2*pi);         % 比例系数Q:Nsz*1
rr = rr*1000;
for ifreq = 1:Nfreq
    
%     offset = ( ifreq - 1 )*( Ndepth*Nsr + Ndepth*Nsr)*4 + origin;
%     fseek(fid, offset, 'bof');
    k_real = fread( fid, Nmodes(ifreq), 'single' );
    k_imag = fread( fid, Nmodes(ifreq), 'single' );
    krm = k_real + 1i*k_imag;  % 水平波数

    phiz = fread( fid, Ndepth*Nmodes(ifreq), 'single' ); % 简正波函数：Ndepth*Nmode
    phiz = reshape(phiz, Ndepth, Nmodes(ifreq) );
    phiSz = phiz(gridDepthIdx, :);  % Nsz*Nmode
    phiRz = phiz(rdIdx, :);         % Nrd*Nmode
    w = nan*zeros(Nsz,Nsr,Nrd);
    for ird = 1:length(rd)
        % 格林函数przw: Nsz*Nsr=[(Nsz*Nmode).*(1*Nmode)]*[(Nmode*Nsr)./(Nmode*Nsr)];
        w(:,:,ird) = Q*(phiSz.*phiRz(ird,:))*(exp(-1i*krm*rr(ird,:))./sqrt(krm*rr(ird,:))); 
    end

    b = sqrt(sum(w.*conj(w), 3));
    w = w./b;                         % 对阵列归一化
    w = reshape(w, Nsz*Nsr, Nrd);
    Beam = Beam + reshape( conj(w)*Rf(:, :, ifreq).*w*ones(Nrd, 1), Nsz, Nsr );
end
Beam = abs(Beam);

Beam = Beam/Nfreq;
% t3 = toc;
fclose(fid);

