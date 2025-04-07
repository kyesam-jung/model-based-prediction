clc
clear
close all

% Connectome relationships (Low-dimensional model fitting)
% --------------------------------------------------------
% X: features [Schaefer_eFCeSC, HarvOxf_eFCeSC, Schaefer_sFCeFC, HarvOxf_sFCeFC]
% y: targets (0: female, 1: male)
% V: brain volumes = cortical surface + subcortical areas + white matter (in cubic centimeter, cc)
X = readmatrix('../data/Feature_for_Classification_N270_LowDim.csv','Delimiter',',');
y = readmatrix('../data/Female_Male_for_Classification_N270_LowDim.csv','Delimiter',',');
V = readmatrix('../data/BrainVolume_for_Classification_N270_LowDim.csv','Delimiter',',');

% PCA
% ---
% COEFF: Loadning for [PC1, PC2, PC3, PC4]
[COEFF,SCORE,LATENT,~,EXPLAINED] = pca(X(:,1:4));

% Results
% -------
test_str = ["Personal_A","Personal_C","Personal_E","Personal_N","Personal_O"];
featureCond = {'EMP','SIM','EMPSIM'};
dimCond = ["LowDim","HighDim"];
for nCond = 1:numel(featureCond)
    for nDimCond = 1:2
        if nCond == 1 && nDimCond == 2
            fprintf('Skip %s, %s\n',featureCond{nCond},dimCond(nDimCond));
        else
            temp_accu_test  = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_ACCURACY_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_accu_train = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_ACCURACY_TRAIN.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA.%s.%s_ACCURACY_TEST = temp_accu_test;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_ACCURACY_TRAIN = temp_accu_train;',featureCond{nCond},dimCond(nDimCond)));
        end
    end
end
for nCond = 1:3 %numel(featureCond)
    for nDimCond = 1:2
        if any([nCond == 1 & nDimCond == 2, nCond == 3 & nDimCond == 2])
            fprintf('Skip %s, %s\n',featureCond{nCond},dimCond(nDimCond));
        else
            temp_r_test     = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_r_train    = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_PEARSON_R_TRAIN.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA.%s.%s_COG_PEARSON_R_TEST = temp_r_test;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_COG_PEARSON_R_TRAIN = temp_r_train;',featureCond{nCond},dimCond(nDimCond)));
        end
    end
end
for nCond = 1:2 % numel(featureCond)
    for nDimCond = 1:2
        if nCond == 1 && nDimCond == 2
                fprintf('Skip %s, %s\n',featureCond{nCond},dimCond(nDimCond));
        else
            for nTest = 1:5
                temp_r_test  = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                temp_r_train = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_PEARSON_R_TRAIN.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                eval(sprintf('DATA.%s.%s_%s_PEARSON_R_TEST = temp_r_test;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
                eval(sprintf('DATA.%s.%s_%s_PEARSON_R_TRAIN = temp_r_train;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
            end
        end
    end
end

acc = [DATA.EMP.LowDim_ACCURACY_TRAIN(:,3),DATA.SIM.LowDim_ACCURACY_TRAIN(:,3),DATA.EMPSIM.LowDim_ACCURACY_TRAIN(:,3),...
       DATA.EMP.LowDim_ACCURACY_TEST(:,3),DATA.SIM.LowDim_ACCURACY_TEST(:,3),DATA.EMPSIM.LowDim_ACCURACY_TEST(:,3)];
pfm = [DATA.EMP.LowDim_COG_PEARSON_R_TRAIN(:,3),DATA.SIM.LowDim_COG_PEARSON_R_TRAIN(:,3),DATA.EMPSIM.LowDim_COG_PEARSON_R_TRAIN(:,3),...
       DATA.EMP.LowDim_COG_PEARSON_R_TEST(:,3),DATA.SIM.LowDim_COG_PEARSON_R_TEST(:,3),DATA.EMPSIM.LowDim_COG_PEARSON_R_TEST(:,3)];
%% Statistics
clc
fprintf("Median of eFC vs. eSC, Schaefer 100P  (Emp.) = %0.4f\n",median(X(:,1)));
fprintf("Median of eFC vs. eSC, Harvard-Oxford (Emp.) = %0.4f\n",median(X(:,2)));
fprintf("Median of eFC vs. sFC, Schaefer 100P  (Sim.) = %0.4f\n",median(X(:,3)));
fprintf("Median of eFC vs. sFC, Harvard-Oxford (Sim.) = %0.4f\n",median(X(:,4)));
fprintf("Mean of eFC vs. eSC, Schaefer 100P  (Emp.) = %0.4f\n",mean(X(:,1)));
fprintf("Mean of eFC vs. eSC, Harvard-Oxford (Emp.) = %0.4f\n",mean(X(:,2)));
fprintf("Mean of eFC vs. sFC, Schaefer 100P  (Sim.) = %0.4f\n",mean(X(:,3)));
fprintf("Mean of eFC vs. sFC, Harvard-Oxford (Sim.) = %0.4f\n",mean(X(:,4)));
fprintf("IQR of eFC vs. eSC, Schaefer 100P  (Emp.) = %0.4f\n",iqr(X(:,1)));
fprintf("IQR of eFC vs. eSC, Harvard-Oxford (Emp.) = %0.4f\n",iqr(X(:,2)));
fprintf("IQR of eFC vs. sFC, Schaefer 100P  (Sim.) = %0.4f\n",iqr(X(:,3)));
fprintf("IQR of eFC vs. sFC, Harvard-Oxford (Sim.) = %0.4f\n",iqr(X(:,4)));
[~,~,s] = ranksum(X(:,1),X(:,2));
es = s.zval/sqrt(size(X,1));
fprintf("Effect size between Emp. Schaefer and Emp. Harvard-Oxford = %0.4f\n",es);
[~,~,s] = ranksum(X(:,3),X(:,4));
es = s.zval/sqrt(size(X,1));
fprintf("Effect size between Sim. Schaefer and Sim. Harvard-Oxford = %0.4f\n",es);
[~,~,s] = ranksum(X(:,1),X(:,3));
es = s.zval/sqrt(size(X,1));
fprintf("Effect size between Emp. Schaefer and Sim. Schaefer = %0.4f\n",es);
[~,~,s] = ranksum(X(:,2),X(:,4));
es = s.zval/sqrt(size(X,1));
fprintf("Effect size between Emp. Harvard-Oxford and Sim. Harvard-Oxford = %0.4f\n",es);
%% Figure 1
N = 20;
M = rand(N,N) - 0.9;
M(M<0) = 0; M = M + M';
L = cell(N,1);
for nRegion = 1:N
    L{nRegion,1} = num2str(nRegion);
end
cmap = repmat([0,0,0],10,1);

% Blank figure
% ------------
fig = figure(1);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','Figure 1: random connectome');
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');

temp_ax = plot_network_circular_plot(M,temp_ax,0,[0,max(M(:))],cmap,L,'symmetric',10);
text(temp_ax,0-0.8,1,'Left','HorizontalAlignment','right','FontSize',10,'FontWeight','Normal');
text(temp_ax,0+0.8,1,'Right','HorizontalAlignment','left','FontSize',10,'FontWeight','Normal');
axis(temp_ax,'off');
temp_ax.Position = [0.10,0.15,0.35,0.7];

cmap = gray(100);
cmap = cmap(end:-1:1,:);
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
imagesc(temp_ax,abs(M));axis(temp_ax,'image','ij');colormap(cmap);axis(temp_ax,'off');
temp_ax.Position = [0.55,0.15,0.35,0.7];

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure1_connectome.pdf');
%% Figure 2
atlList = {'Schaefer','Harvard-Oxford'};
conList = {'eFC vs. eSC','eFC vs. sFC'};

% Blank figure
% ------------
fig = figure(2);clf;set(gcf,'Color','w','Position',[1,1,500,500],'Name','Figure 2: Distributions and PCA');

% Figure 2a
% ---------
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
hp = nan(numel(conList)*numel(atlList),1);
nPlot = 0;
cmap = [0.2, 0.2, 0.8; ...
        0.8, 0.2, 0.2; ...
        [0.0, 0.3, 0.9]*0.7; ...
        [0.9, 0.3, 0.0]*0.7];
for nCon = 1:numel(conList)
    if nCon == 1
        lw = 0.5;
    elseif nCon == 2
        lw = 2.0;
    end
    for nAtl = 1:numel(atlList)
        nPlot = nPlot + 1;
        [N,EDGES,BIN] = histcounts(X(:,nPlot),-1:0.05:1);
        x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
        xq = -1:0.05/10:1;
        vq = interp1(x,N,xq,'pchip');
        temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(nPlot,:),'LineWidth',lw);
        temp_hp.FaceAlpha = 0.2;
        temp_hp.EdgeColor = cmap(nPlot,:);
        temp_hp.EdgeAlpha = 0.5;
        hp(nPlot) = temp_hp;
    end
end
hl = legend(hp,{'eFC vs. eSC, Schaefer 100P','eFC vs. eSC, Harvard-Oxford','eFC vs. sFC, Schaefer 100P','eFC vs. sFC, Harvard-Oxford'},'box','off','FontWeight','normal','FontSize',12);
hl.Position = [0.55,0.78,0.2,0.12];
xlim(temp_ax,[0.0,1.0]);
set(temp_ax,'FontWeight','normal','FontSize',12);
xlabel(temp_ax,'Connectome relationship (Pearson''s correlation)','FontWeight','normal','FontSize',14);
ylabel(temp_ax,'Number of subjects','FontWeight','normal','FontSize',14);
title(temp_ax,'Profiles of features','FontWeight','normal','FontSize',14);
temp_ax.Position = [0.14,0.67,0.80,0.25];
annotation(fig,'textbox', [0.04,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'line', [0.845,0.86],[0.888,0.888], 'color', 'k', 'linewidth', 1);
annotation(fig,'line', [0.850,0.86],[0.855,0.855], 'color', 'k', 'linewidth', 1);
annotation(fig,'line', [0.860,0.86],[0.888,0.855], 'color', 'k', 'linewidth', 1);
annotation(fig,'textbox',[0.859,0.866,0.01,0.01],'string','Emp.','edgecolor','none','fontsize',14,'fontweight','n','horizontalalignment','left','VerticalAlignment','middle');

annotation(fig,'line', [0.845,0.86],[0.888,0.888]-0.06, 'color', 'k', 'linewidth', 1);
annotation(fig,'line', [0.850,0.86],[0.855,0.855]-0.06, 'color', 'k', 'linewidth', 1);
annotation(fig,'line', [0.860,0.86],[0.888,0.855]-0.06, 'color', 'k', 'linewidth', 1);
annotation(fig,'textbox',[0.859,0.866-0.06,0.01,0.01],'string','Sim.','edgecolor','none','fontsize',14,'fontweight','n','horizontalalignment','left','VerticalAlignment','middle');

% Figure 2b
% ---------
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
set(temp_ax,'XTick',1:4,'XTickLabel',{'PC1','PC2','PC3','PC4'},'YTick',-1:0.25:1,'FontWeight','normal','FontSize',12,'box','on');
grid(temp_ax,'on');
nPlot = 0;
bw = 0.18;
for nCon = 1:numel(conList)
    if nCon == 1
        lw = 0.5;
    elseif nCon == 2
        lw = 2.0;
    end
    for nAtl = 1:numel(atlList)
        nPlot = nPlot + 1;
        bar(temp_ax,1-bw*1.5+(nPlot-1)*bw,COEFF(nPlot,1),'FaceColor',cmap(nPlot,:),'EdgeColor',cmap(nPlot,:),'FaceAlpha',0.2,'EdgeAlpha',0.5,'LineWidth',lw,'BarWidth',bw);
        bar(temp_ax,2-bw*1.5+(nPlot-1)*bw,COEFF(nPlot,2),'FaceColor',cmap(nPlot,:),'EdgeColor',cmap(nPlot,:),'FaceAlpha',0.2,'EdgeAlpha',0.5,'LineWidth',lw,'BarWidth',bw);
        bar(temp_ax,3-bw*1.5+(nPlot-1)*bw,COEFF(nPlot,3),'FaceColor',cmap(nPlot,:),'EdgeColor',cmap(nPlot,:),'FaceAlpha',0.2,'EdgeAlpha',0.5,'LineWidth',lw,'BarWidth',bw);
        bar(temp_ax,4-bw*1.5+(nPlot-1)*bw,COEFF(nPlot,4),'FaceColor',cmap(nPlot,:),'EdgeColor',cmap(nPlot,:),'FaceAlpha',0.2,'EdgeAlpha',0.5,'LineWidth',lw,'BarWidth',bw);
    end
end
xlim(temp_ax,[0.4,4.6]);
ylim(temp_ax,[-1,1]);
xlabel(temp_ax,'PCs','FontWeight','normal','FontSize',14);
ylabel(temp_ax,'PC loadings','FontWeight','normal','FontSize',14);
temp_ax.Position = [0.14,0.13,0.32,0.36];
annotation(fig,'textbox', [0.04,0.55,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Figure 2c
% ---------
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
C_EXPLAINED = cumsum(EXPLAINED);
hb0 = bar(temp_ax,1:4,C_EXPLAINED,'FaceColor',[0.5,1.0,0]*0.6,'FaceAlpha',0.2,'EdgeColor',[0.5,1.0,0]*0.5);
hb = [];
for nPC = 1:size(COEFF,2)
    lgc = false(size(COEFF,2),1);
    lgc(1:nPC) = true;
    X_ = SCORE(:,lgc)*COEFF(:,lgc)';
    for nPlot = 1:4
        if nPlot <= 2
            lw = 0.5;
        else
            lw = 2.0;
        end
        r = corr(X(:,nPlot),X_(:,nPlot),'Type','Pearson');
        bh(nPlot) = bar(temp_ax,nPC-bw*1.5+(nPlot-1)*bw,r*r*100,'FaceColor',cmap(nPlot,:),'EdgeColor',cmap(nPlot,:),'FaceAlpha',0.2,'EdgeAlpha',0.5,'LineWidth',lw,'BarWidth',bw);
    end
end
xlim(temp_ax,[0.4,4.6]);
set(temp_ax,'XTick',1:4,'YTick',0:10:100,'FontWeight','normal','FontSize',12,'box','on');
grid(temp_ax,'on');
xlabel(temp_ax,'Number of PCs','FontWeight','normal','FontSize',14);
ylabel(temp_ax,'Explained variance (%)','FontWeight','normal','FontSize',14);
legend([hb0],{'Total'},'Location','southeast');
temp_ax.Position = [0.62,0.13,0.30,0.36];
annotation(fig,'textbox', [0.52,0.55,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure2.pdf');
%% Figure 3

% Blank figure
% ------------
fig = figure(3);clf;set(gcf,'Color','w','Position',[1,1,500,500],'Name','Figure 3: Training and Testing');

% Figure 3a
% ---------
cmap = lines(3);
condList = {'Emp.','Sim.',sprintf('Merged\n(Emp. & Sim.)')};
for nCond = 1:3
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');

    % Train
    [N,EDGES] = histcounts(acc(:,nCond),0.4:0.02:0.8);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    hb1 = bar(temp_ax,x,y,'FaceColor',cmap(2,:),'FaceAlpha',0.1,'EdgeColor','none');
    xq = 0.4:0.01/10:0.8;
    vq = interp1(x,N,xq,'pchip');
    temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(2,:),'LineWidth',1.0);
    temp_hp.FaceAlpha = 0.2;
    temp_hp.EdgeColor = cmap(2,:);
    temp_hp.EdgeAlpha = 0.5;
    hb1 = temp_hp;
    
    % Test
    [N,EDGES] = histcounts(acc(:,nCond+3),0.4:0.02:0.8);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    hb2 = bar(temp_ax,x,y,'FaceColor',cmap(1,:),'FaceAlpha',0.1,'EdgeColor','none');
    xq = 0.4:0.01/10:0.8;
    vq = interp1(x,N,xq,'pchip');
    temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(1,:),'LineWidth',1.0);
    temp_hp.FaceAlpha = 0.2;
    temp_hp.EdgeColor = cmap(1,:);
    temp_hp.EdgeAlpha = 0.5;
    hb2 = temp_hp;

    yl = temp_ax.YLim;
    hl1 = line(temp_ax,[1,1]*mean(acc(:,nCond)),yl,'Color',cmap(2,:),'LineStyle',':','LineWidth',2);
    hl2 = line(temp_ax,[1,1]*mean(acc(:,nCond+3)),yl,'Color',cmap(1,:),'LineStyle',':','LineWidth',2);
    hl3 = line(temp_ax,[1,1]*-1,yl*-1,'Color','k','LineStyle',':','LineWidth',2);
    set(temp_ax,'XTickLabel','');
    set(temp_ax,'YTick',[0,yl(2)],'YTickLabel',[0,1]);
    if nCond == 1
        title(temp_ax,'Classification','FontWeight','normal','FontSize',14);
    elseif nCond == 2
        hl = legend(temp_ax,[hb1,hb2,hl3],{'Train','Test','Means'},'FontSize',12,'FontWeight','n','box','off');
        hl.Position = [0.42,0.73,0.08,0.10];
        ylabel(temp_ax,'Number of folds (scaled)','FontWeight','normal','FontSize',14);
    elseif nCond == 3
        set(temp_ax,'XTick',0.4:0.1:0.8,'XTickLabel',0.4:0.1:0.8,'FontWeight','normal','FontSize',12);
        xlabel(temp_ax,'Balanced accuracy','FontWeight','normal','FontSize',14);
    end
    xlim(temp_ax,[0.4,0.8]);
    ylim(temp_ax,[0,yl(2)]);
    text(temp_ax,0.41,yl(2)*0.99,condList{nCond},'FontWeight','normal','FontSize',13,'verticalalignment','top');
    temp_ax.Position = [0.10,0.85-0.12*(nCond-1),0.40,0.08];
end
annotation(fig,'textbox', [0.03,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Figure 3b
% ---------
for nCond = 1:3
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    
    % Train
    [N,EDGES] = histcounts(pfm(:,nCond),-0.4:0.02:0.8);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    hb1 = bar(temp_ax,x,y,'FaceColor',cmap(2,:),'FaceAlpha',0.1,'EdgeColor','none');
    xq = -0.4:0.01/10:0.4;
    vq = interp1(x,N,xq,'pchip');
    temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(2,:),'LineWidth',1.0);
    temp_hp.FaceAlpha = 0.2;
    temp_hp.EdgeColor = cmap(2,:);
    temp_hp.EdgeAlpha = 0.5;
    hb1 = temp_hp;
    
    % Test
    [N,EDGES] = histcounts(pfm(:,nCond+3),-0.4:0.02:0.8);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    hb2 = bar(temp_ax,x,y,'FaceColor',cmap(1,:),'FaceAlpha',0.1,'EdgeColor','none');
    xq = -0.4:0.01/10:0.8;
    vq = interp1(x,N,xq,'pchip');
    temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(1,:),'LineWidth',1.0);
    temp_hp.FaceAlpha = 0.2;
    temp_hp.EdgeColor = cmap(1,:);
    temp_hp.EdgeAlpha = 0.5;
    hb2 = temp_hp;

    yl = temp_ax.YLim;
    hl0 = line(temp_ax,[1,1]*0,yl,'Color',[1,1,1]*0.5,'LineStyle','-','LineWidth',0.5);
    hl1 = line(temp_ax,[1,1]*mean(pfm(:,nCond)),yl,'Color',cmap(2,:),'LineStyle',':','LineWidth',2);
    hl2 = line(temp_ax,[1,1]*mean(pfm(:,nCond+3)),yl,'Color',cmap(1,:),'LineStyle',':','LineWidth',2);
    set(temp_ax,'XTickLabel','');
    set(temp_ax,'YTick',[0,yl(2)],'YTickLabel',[0,1],'FontWeight','normal','FontSize',12);
    if nCond == 1
        hl = legend(temp_ax,[hb1,hb2,hl3],{'Train','Test','Means'},'FontSize',12,'FontWeight','n','box','off');
        hl.Position = [0.42,0.39,0.08,0.10];
        title(temp_ax,'Prediction','FontWeight','normal','FontSize',14);
    elseif nCond == 2
        ylabel(temp_ax,'Number of folds (scaled)','FontWeight','normal','FontSize',14);
    elseif nCond == 3
        set(temp_ax,'XTick',-0.4:0.2:0.6,'XTickLabel',{'-0.4','-0.2','0.0','0.2','0.4','0.6'},'FontWeight','normal','FontSize',12,'XTickLabelRotation',0);
        xlabel(temp_ax,'Pearson''s correlation','FontWeight','normal','FontSize',14);
    end
    xlim(temp_ax,[-0.4,0.6]);
    ylim(temp_ax,[0,yl(2)]);
    text(temp_ax,-0.39,yl(2)*0.99,condList{nCond},'FontWeight','normal','FontSize',13,'verticalalignment','top');
    temp_ax.Position = [0.10,0.75-0.12*(nCond-1+3),0.40,0.08];
end
annotation(fig,'textbox', [0.03,0.52,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Figure 3c
% ---------
condList = {'Emp.','Sim.',sprintf('Merged')};
cmap = lines(5);
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
for nCond = 1:6
    eval(sprintf('Y%i = acc(:,%i);',nCond,nCond));
end
l = numel(Y1);
d = 0.2;
[p1,~,s] = ranksum(Y1,Y2); es1 = s.zval/sqrt(l);
[p2,~,s] = ranksum(Y2,Y3); es2 = s.zval/sqrt(l);
[p3,~,s] = ranksum(Y1,Y3); es3 = s.zval/sqrt(l);
[p4,~,s] = ranksum(Y4,Y5); es4 = s.zval/sqrt(l);
[p5,~,s] = ranksum(Y5,Y6); es5 = s.zval/sqrt(l);
[p6,~,s] = ranksum(Y4,Y6); es6 = s.zval/sqrt(l);
[x] = ksj_violin_scatter(Y4,1,13,0.40,0);
hsp1 = scatter(temp_ax,x,Y4,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
[x] = ksj_violin_scatter(Y5,2,13,0.40,0);
hsp2 = scatter(temp_ax,x,Y5,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
[x] = ksj_violin_scatter(Y6,3,13,0.40,0);
hsp3 = scatter(temp_ax,x,Y6,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
hbp1 = boxplot(temp_ax,Y4,'Positions',1,'Orientation','vertical','Width',0.2,'Colors',cmap(3,:)*0);
hbp2 = boxplot(temp_ax,Y5,'Positions',2,'Orientation','vertical','Width',0.2,'Colors',cmap(4,:)*0);
hbp3 = boxplot(temp_ax,Y6,'Positions',3,'Orientation','vertical','Width',0.2,'Colors',cmap(5,:)*0);
set(hbp1,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
set(hbp2,{'linew'},{1});set(hbp2(6),'Color',[0,0,0]);
set(hbp3,{'linew'},{1});set(hbp3(6),'Color',[0,0,0]);
plot(temp_ax,[-0.1,0.1]+1,[1,1]*quantile(Y4,0.5),'Color','k','LineWidth',3);
plot(temp_ax,[-0.1,0.1]+2,[1,1]*quantile(Y5,0.5),'Color','k','LineWidth',3);
plot(temp_ax,[-0.1,0.1]+3,[1,1]*quantile(Y6,0.5),'Color','k','LineWidth',3);
ylim(temp_ax,[0.35,0.85]);
gap = (temp_ax.YLim(2) - temp_ax.YLim(1))*0.1;
M0  = temp_ax.YLim(2) - gap*0.2;
if p4 < 0.05^3
    M = M0 - gap*0.9;
    line(temp_ax,[1,1,2,2],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
if p5 < 0.05^3
    M = M0 - gap*0.6;
    line(temp_ax,[2,2,3,3],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
if p6 < 0.05^3
    M = M0 - gap*0.3;
    line(temp_ax,[1,1,3,3],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
xlim(temp_ax,[0.5,3.5]);
ylim(temp_ax,[0.35,0.85]);
set(temp_ax,'FontSize',12,'FontWeight','n');
set(temp_ax,'YTick',0.4:0.1:0.9);
set(temp_ax,'XTick',[1,2,3],'XTickLabel',condList,'FontSize',12,'FontWeight','n','XTickLabelRotation',0);
grid(temp_ax,'on');
ylabel(temp_ax,'Balanced accuracy');
title(temp_ax,'Classification','FontWeight','normal','FontSize',14);
temp_ax.Position = [0.67,0.61,0.28,0.32];
annotation(fig,'textbox', [0.58,0.98,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Figure 3d
% ---------
cmap = lines(5);
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
for nCond = 1:6
    eval(sprintf('Y%i = pfm(:,%i);',nCond,nCond));
end
l = numel(Y1);
d = 0.2;
[p1,~,s] = ranksum(Y1,Y2); es1 = s.zval/sqrt(l);
[p2,~,s] = ranksum(Y2,Y3); es2 = s.zval/sqrt(l);
[p3,~,s] = ranksum(Y1,Y3); es3 = s.zval/sqrt(l);
[p4,~,s] = ranksum(Y4,Y5); es4 = s.zval/sqrt(l);
[p5,~,s] = ranksum(Y5,Y6); es5 = s.zval/sqrt(l);
[p6,~,s] = ranksum(Y4,Y6); es6 = s.zval/sqrt(l);
[x] = ksj_violin_scatter(Y4,1,13,0.40,0);
hsp1 = scatter(temp_ax,x,Y4,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
[x] = ksj_violin_scatter(Y5,2,13,0.40,0);
hsp2 = scatter(temp_ax,x,Y5,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
[x] = ksj_violin_scatter(Y6,3,13,0.40,0);
hsp3 = scatter(temp_ax,x,Y6,15,'Marker','o','MarkerFaceColor',[1,1,1]*0.3,'MarkerEdgeColor','none','MarkerFaceAlpha',0.1);
hbp1 = boxplot(temp_ax,Y4,'Positions',1,'Orientation','vertical','Width',0.2,'Colors',cmap(3,:)*0);
hbp2 = boxplot(temp_ax,Y5,'Positions',2,'Orientation','vertical','Width',0.2,'Colors',cmap(4,:)*0);
hbp3 = boxplot(temp_ax,Y6,'Positions',3,'Orientation','vertical','Width',0.2,'Colors',cmap(5,:)*0);
set(hbp1,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
set(hbp2,{'linew'},{1});set(hbp2(6),'Color',[0,0,0]);
set(hbp3,{'linew'},{1});set(hbp3(6),'Color',[0,0,0]);
plot(temp_ax,[-0.1,0.1]+1,[1,1]*quantile(Y4,0.5),'Color','k','LineWidth',3);
plot(temp_ax,[-0.1,0.1]+2,[1,1]*quantile(Y5,0.5),'Color','k','LineWidth',3);
plot(temp_ax,[-0.1,0.1]+3,[1,1]*quantile(Y6,0.5),'Color','k','LineWidth',3);
ylim(temp_ax,[-0.55,0.65]);
gap = (temp_ax.YLim(2) - temp_ax.YLim(1))*0.1;
M0  = temp_ax.YLim(2) - gap*0.2;
if p4 < 0.05^3
    M = M0 - gap*0.6;
    line(temp_ax,[1,1,2,2],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
if p5 < 0.05^3
    M = M0 - gap*0.9;
    line(temp_ax,[2,2,3,3],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
if p6 < 0.05^3
    M = M0 - gap*0.3;
    line(temp_ax,[1,1,3,3],[M-gap*0.2,M,M,M-gap*0.2],'Color',[0.6,0,1]);
end
xlim(temp_ax,[0.5,3.5]);
ylim(temp_ax,[-0.55,0.65]);
set(temp_ax,'FontSize',12,'FontWeight','n');
set(temp_ax,'YTick',-0.6:0.2:0.8);
set(temp_ax,'XTick',[1,2,3],'XTickLabel',condList,'FontSize',12,'FontWeight','n','XTickLabelRotation',0);
grid(temp_ax,'on');
ylabel(temp_ax,'Pearson''s correlation');
title(temp_ax,'Prediction','FontWeight','normal','FontSize',14);
temp_ax.Position = [0.67,0.15,0.28,0.32];
annotation(fig,'textbox', [0.58,0.52,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure3.pdf');
%% Figure 4
clc
plotCond = ["Emp.","Sim. (Low dim.)","Sim. (High dim.)"];
test_str = ["Agreeableness","Conscientiousness","Extraversion","Neuroticism","Openness"];
cmap = lines(5);

% Blank figure
% ------------
fig = figure(4);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','Figure 4');

% a: Classification
% -----------------
Y = [DATA.EMP.LowDim_ACCURACY_TEST(:,3),...
     DATA.SIM.LowDim_ACCURACY_TEST(:,3),...
     DATA.SIM.HighDim_ACCURACY_TEST(:,3)];

temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
hp = nan(5,1);
MY = 0;
for nPlot = 1:numel(plotCond)
    y = Y(:,nPlot);
    [N,EDGES] = histcounts(y,0.0:0.05:1.0);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    xq = 0.4:0.001:0.8;
    vq = interp1(x,N,xq,'pchip');
    hp(nPlot) = plot(temp_ax,xq,vq,'Color',cmap(nPlot,:),'LineWidth',2);
    if max(y) > MY
        MY = round((max(y)*1.1)/50)*50;
    end
end
for nPlot = 1:numel(plotCond)
    line(temp_ax,[1,1]*mean(Y(:,nPlot)),[0.0,MY],'Color',cmap(nPlot,:),'LineStyle','-.','LineWidth',2);
end
hp(6) = line(temp_ax,[0,0],[0,0],'Color','k','LineStyle','-.','LineWidth',2);
ylim(temp_ax,[0.0,MY]);
xlim(temp_ax,[0.4,0.8]);
xlabel(temp_ax,'Balanced accuracy','FontSize',12,'FontWeight','n');
ylabel(temp_ax,'Frequency','FontSize',12,'FontWeight','n');
title(temp_ax,'Sex classification','FontSize',12,'FontWeight','n');
set(temp_ax,'XTick',0.4:0.1:0.8,'FontSize',12,'FontWeight','n');
grid(temp_ax,'on');
temp_ax.Position = [0.07,0.63,0.3,0.3];

% b: Prediction of cognition
% --------------------------
Y = [DATA.EMP.LowDim_COG_PEARSON_R_TEST(:,3),...
     DATA.SIM.LowDim_COG_PEARSON_R_TEST(:,3),...
     DATA.SIM.HighDim_COG_PEARSON_R_TEST(:,3)];

temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
hp = nan(numel(plotCond),1);
MY = 0;
for nPlot = 1:numel(plotCond)
    y = Y(:,nPlot);
    [N,EDGES] = histcounts(y,-1.0:0.15:1.0);
    x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
    y = N;
    xq = -1.0:0.001:1.0;
    vq = interp1(x,N,xq,'pchip');
    hp(nPlot) = plot(temp_ax,xq,vq,'Color',cmap(nPlot,:),'LineWidth',2);
    if max(y) > MY
        MY = round((max(y)*1.1)/50)*50;
    end
end
for nPlot = 1:numel(plotCond)
    line(temp_ax,[1,1]*mean(Y(:,nPlot)),[0.0,MY],'Color',cmap(nPlot,:),'LineStyle','-.','LineWidth',2);
end
hp(numel(plotCond)+1) = line(temp_ax,[0,0],[0,0],'Color','k','LineStyle','-.','LineWidth',2);
ylim(temp_ax,[0.0,MY]);
xlim(temp_ax,[-0.6,0.6]);
xlabel(temp_ax,'Pearson''s correlation','FontSize',12,'FontWeight','n');
ylabel(temp_ax,'Frequency','FontSize',12,'FontWeight','n');
title(temp_ax,'Prediction of cognition','FontSize',12,'FontWeight','n');
set(temp_ax,'XTick',-1.0:0.2:1.0,'FontSize',12,'FontWeight','n');
grid(temp_ax,'on');
hl = legend(temp_ax,hp,[plotCond,'Means'],'FontSize',14,'FontWeight','n');
hl.Position = [0.43,0.45,hl.Position(3)*1.2,hl.Position(4)*1.2];
hl.Box = "off";
temp_ax.Position = [0.07,0.13,0.3,0.3];

% c: Prediction of personality
% ----------------------------
for nCase = 1:5
    switch nCase
        case 1
            Y = [DATA.EMP.LowDim_Personal_A_PEARSON_R_TEST(:,3),...
                 DATA.SIM.LowDim_Personal_A_PEARSON_R_TEST(:,3),...
                 DATA.SIM.HighDim_Personal_A_PEARSON_R_TEST(:,3)];
        case 2
            Y = [DATA.EMP.LowDim_Personal_C_PEARSON_R_TEST(:,3),...
                 DATA.SIM.LowDim_Personal_C_PEARSON_R_TEST(:,3),...
                 DATA.SIM.HighDim_Personal_C_PEARSON_R_TEST(:,3)];
        case 3
            Y = [DATA.EMP.LowDim_Personal_E_PEARSON_R_TEST(:,3),...
                 DATA.SIM.LowDim_Personal_E_PEARSON_R_TEST(:,3),...
                 DATA.SIM.HighDim_Personal_E_PEARSON_R_TEST(:,3)];
        case 4
            Y = [DATA.EMP.LowDim_Personal_N_PEARSON_R_TEST(:,3),...
                 DATA.SIM.LowDim_Personal_N_PEARSON_R_TEST(:,3),...
                 DATA.SIM.HighDim_Personal_N_PEARSON_R_TEST(:,3)];
        case 5
            Y = [DATA.EMP.LowDim_Personal_O_PEARSON_R_TEST(:,3),...
                 DATA.SIM.LowDim_Personal_O_PEARSON_R_TEST(:,3),...
                 DATA.SIM.HighDim_Personal_O_PEARSON_R_TEST(:,3)];
    end
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    hp = nan(numel(plotCond),1);
    MY = 0;
    for nPlot = 1:numel(plotCond)
        y = Y(:,nPlot);
        [N,EDGES] = histcounts(y,-1.0:0.15:1.0);
        x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
        y = N;
        xq = -1.0:0.001:1.0;
        vq = interp1(x,N,xq,'pchip');
        hp(nPlot) = plot(temp_ax,xq,vq,'Color',cmap(nPlot,:),'LineWidth',2);
        if max(y) > MY
            MY = round((max(y)*1.1)/50)*50;
        end
    end
    for nPlot = 1:numel(plotCond)
        line(temp_ax,[1,1]*mean(Y(:,nPlot)),[0.0,MY],'Color',cmap(nPlot,:),'LineStyle','-.','LineWidth',2);
    end
    hp(numel(plotCond)+1) = line(temp_ax,[0,0],[0,0],'Color','k','LineStyle','-.','LineWidth',2);
    ylim(temp_ax,[0.0,MY]);
    xlim(temp_ax,[-0.6,0.6]);
    if any(nCase == [1,4,5])
        xlabel(temp_ax,'Pearson''s correlation','FontSize',12,'FontWeight','n');
    end
    ylabel(temp_ax,'Frequency','FontSize',12,'FontWeight','n');
    title(temp_ax,test_str{nCase},'FontSize',12,'FontWeight','n','Interpreter','none');
    set(temp_ax,'XTick',-1.0:0.2:1.0,'FontSize',12,'FontWeight','n','YTick',0:50:500,'YTickLabel',[]);
    grid(temp_ax,'on');
    if nCase == 1
        temp_ax.Position = [0.45,0.73,0.20,0.20];
    elseif nCase == 2
        temp_ax.Position = [0.74,0.73,0.20,0.20];
    elseif nCase == 3
        temp_ax.Position = [0.74,0.43,0.20,0.20];
    elseif nCase == 4
        temp_ax.Position = [0.45,0.13,0.20,0.20];
    elseif nCase == 5
        temp_ax.Position = [0.74,0.13,0.20,0.20];
    end
end

annotation(fig,'textbox', [0.02,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.02,0.48,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.43,0.98,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.72,0.98,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.72,0.68,0.01,0.01],'string','e','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.43,0.38,0.01,0.01],'string','f','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.72,0.38,0.01,0.01],'string','g','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure4.pdf');
%% Figure 5
plotCond = ["Emp.","Sim. (Low dim.)","Sim. (High dim.)"];
cmap = lines(5);

% Blank figure
% ------------
fig = figure(5);clf;set(gcf,'Color','w','Position',[1,1,500,250],'Name','Figure 5: Overall prediction of the big five');

Y_ALL = [];
Y = [DATA.EMP.LowDim_Personal_A_PEARSON_R_TEST(:,3);...
     DATA.EMP.LowDim_Personal_C_PEARSON_R_TEST(:,3);...
     DATA.EMP.LowDim_Personal_E_PEARSON_R_TEST(:,3);...
     DATA.EMP.LowDim_Personal_N_PEARSON_R_TEST(:,3);...
     DATA.EMP.LowDim_Personal_O_PEARSON_R_TEST(:,3)];

Y_ALL = [Y_ALL,Y];

Y = [DATA.SIM.LowDim_Personal_A_PEARSON_R_TEST(:,3);...
     DATA.SIM.LowDim_Personal_C_PEARSON_R_TEST(:,3);...
     DATA.SIM.LowDim_Personal_E_PEARSON_R_TEST(:,3);...
     DATA.SIM.LowDim_Personal_N_PEARSON_R_TEST(:,3);...
     DATA.SIM.LowDim_Personal_O_PEARSON_R_TEST(:,3)];

Y_ALL = [Y_ALL,Y];

Y = [DATA.SIM.HighDim_Personal_A_PEARSON_R_TEST(:,3);...
     DATA.SIM.HighDim_Personal_C_PEARSON_R_TEST(:,3);...
     DATA.SIM.HighDim_Personal_E_PEARSON_R_TEST(:,3);...
     DATA.SIM.HighDim_Personal_N_PEARSON_R_TEST(:,3);...
     DATA.SIM.HighDim_Personal_O_PEARSON_R_TEST(:,3)];

Y_ALL = [Y_ALL,Y];

temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
line(temp_ax,[0,6],[0.0,0.0],'Color','k','LineStyle',':');
for nPlot = 1:numel(plotCond)
    y = Y_ALL(:,nPlot);
    [x] = ksj_violin_scatter(y,nPlot,40,0.40,0);
    temp_hsp = scatter(temp_ax,x,y,10,'Marker','o','MarkerFaceColor','none','MarkerEdgeColor',cmap(nPlot,:)*0.9,'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
    line(temp_ax,[-0.3,0.3]+nPlot,[1,1]*mean(y),'Color',cmap(nPlot,:)*0.7,'LineWidth',2);
end
hsp = nan(numel(plotCond),1);
for nPlot = 1:numel(plotCond)
    temp_hsp = scatter(temp_ax,-1,01,10,'Marker','o','MarkerFaceColor','none','MarkerEdgeColor',cmap(nPlot,:)*0.9,'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',1,'LineWidth',2);
    hsp(nPlot) = temp_hsp;
end
plot(temp_ax,[-0.3,0.3]+nPlot,[1,1]*-1,'Color','k','LineWidth',2);
ylim(temp_ax,[-0.4,0.4]);
xlim(temp_ax,[0,numel(plotCond)+1]);
ylabel(temp_ax,'Pearson''s correlation','FontSize',12,'FontWeight','n');
xlabel(temp_ax,'Feature conditions','FontSize',12,'FontWeight','n');
title(temp_ax,'Prediction of personality (Big-Five total)','FontSize',12,'FontWeight','n');
set(temp_ax,'XTick',1:numel(plotCond),'XTickLabel',[],'XTickLabelRotation',0,'FontSize',12,'FontWeight','n','YTick',-1:0.1:1);
grid(temp_ax,'on');
hl = legend(temp_ax,hsp,plotCond,'FontSize',12,'FontWeight','n');
hl.Position = [0.64,0.46,hl.Position(3),hl.Position(4)];
hl.Box = "off";
temp_ax.Position = [0.12,0.18,0.5,0.7];
annotation(fig,'line', [0.663,0.693],[0.435,0.435], 'color', 'k', 'linewidth', 2);
annotation(fig,'textbox',[0.704,0.428,0.01,0.01],'string','Means','edgecolor','none','fontsize',12,'fontweight','n','horizontalalignment','left','VerticalAlignment','middle');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure5.pdf');
%% Figure 6
colors = load('./RdBu_colormap.txt');

% Predictors
% ----------
F = [];
F.high_p = readmatrix('../data/Feature_for_Personality_N269_HighDim.csv','Delimiter',',');
F.low_p  = readmatrix('../data/Feature_for_Personality_N269_LowDim.csv','Delimiter',',');
F.high_i = readmatrix('../data/Feature_for_CogTotalComp_Unadj_N268_HighDim.csv','Delimiter',',');
F.low_i  = readmatrix('../data/Feature_for_CogTotalComp_Unadj_N268_LowDim.csv','Delimiter',',');
F.high_c = readmatrix('../data/Feature_for_Classification_N270_HighDim.csv','Delimiter',',');
F.low_c  = readmatrix('../data/Feature_for_Classification_N270_LowDim.csv','Delimiter',',');

% SHAP values
% -----------
DATA_SHAP = [];
test_str = ["Personal_A","Personal_C","Personal_E","Personal_N","Personal_O"];
featureCond = {'EMP','SIM','SIM'};
dimCond = ["LowDim","LowDim","HighDim"];
for nCond = 1:3
    for nTest = 1:5
        temp_r_beta    = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_BETA.txt',dimCond(nCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
        temp_idx       = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_CV_TRAIN_INDEX.txt',dimCond(nCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
        temp_r         = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_PEARSON_R_TEST.txt',dimCond(nCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
        temp_shap      = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_SHAP_SUBJECT.txt',dimCond(nCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
        temp_shap_base = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_SHAP_BASE.txt',dimCond(nCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
        eval(sprintf('DATA_SHAP.%s.%s_%s_BETA = temp_r_beta;',featureCond{nCond},dimCond(nCond),test_str(nTest)));
        eval(sprintf('DATA_SHAP.%s.%s_%s_INDEX = temp_idx;',featureCond{nCond},dimCond(nCond),test_str(nTest)));
        eval(sprintf('DATA_SHAP.%s.%s_%s_R_TEST = temp_r;',featureCond{nCond},dimCond(nCond),test_str(nTest)));
        eval(sprintf('DATA_SHAP.%s.%s_%s_SHAP = temp_shap;',featureCond{nCond},dimCond(nCond),test_str(nTest)));
        eval(sprintf('DATA_SHAP.%s.%s_%s_SHAP_BASE = temp_shap_base;',featureCond{nCond},dimCond(nCond),test_str(nTest)));
    end
    temp_r_beta    = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_BETA.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_idx       = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_CV_TRAIN_INDEX.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_r         = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_PEARSON_R_TEST.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_shap      = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_SHAP_SUBJECT.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_shap_base = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_SHAP_BASE.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    eval(sprintf('DATA_SHAP.%s.%s_COG_BETA = temp_r_beta;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_COG_INDEX = temp_idx;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_COG_R_TEST = temp_r;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_COG_SHAP = temp_shap;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_COG_SHAP_BASE = temp_shap_base;',featureCond{nCond},dimCond(nCond)));

    temp_r_beta    = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_BETA.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_idx       = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_CV_TRAIN_INDEX.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_accu      = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_ACCURACY_TEST.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_shap      = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_SHAP_SUBJECT.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    temp_shap_base = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_SHAP_BASE.txt',dimCond(nCond),featureCond{nCond}),'Delimiter',' ');
    eval(sprintf('DATA_SHAP.%s.%s_Classification_BETA = temp_r_beta;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_Classification_INDEX = temp_idx;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_Classification_ACCU_TEST = temp_accu;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_Classification_SHAP = temp_shap;',featureCond{nCond},dimCond(nCond)));
    eval(sprintf('DATA_SHAP.%s.%s_Classification_SHAP_BASE = temp_shap_base;',featureCond{nCond},dimCond(nCond)));
end
clear temp* dimCond featureCond nCond nTest test_str
%% SHAP, beeswarm
D = [];
% mean(DATA_SHAP.EMP.LowDim_Classification_ACCU_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_Classification_ACCU_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Classification_ACCU_TEST(:,3)) -> best
D{7}.SHAP = DATA_SHAP.SIM.HighDim_Classification_SHAP;
D{7}.SHAP_BASE = DATA_SHAP.SIM.HighDim_Classification_SHAP_BASE;
D{7}.str = "Sim. (High dim.)";
D{7}.target = "Sex classification";
D{7}.feature = F.high_c(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_COG_R_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_COG_R_TEST(:,3)) -> best
% mean(DATA_SHAP.SIM.HighDim_COG_R_TEST(:,3))
D{6}.SHAP = DATA_SHAP.SIM.LowDim_COG_SHAP;
D{6}.SHAP_BASE = DATA_SHAP.SIM.LowDim_COG_SHAP_BASE;
D{6}.str = "Sim. (Low dim.)";
D{6}.target = "Cognition";
D{6}.feature = F.low_i(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_Personal_A_R_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_Personal_A_R_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Personal_A_R_TEST(:,3)) -> best
D{5}.SHAP = DATA_SHAP.SIM.HighDim_Personal_A_SHAP;
D{5}.SHAP_BASE = DATA_SHAP.SIM.HighDim_Personal_A_SHAP_BASE;
D{5}.str = "Sim. (High dim.)";
D{5}.target = "Agreeableness";
D{5}.feature = F.high_p(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_Personal_C_R_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_Personal_C_R_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Personal_C_R_TEST(:,3)) -> best
D{4}.SHAP = DATA_SHAP.SIM.HighDim_Personal_C_SHAP;
D{4}.SHAP_BASE = DATA_SHAP.SIM.HighDim_Personal_C_SHAP_BASE;
D{4}.str = "Sim. (High dim.)";
D{4}.target = "Conscientiousness";
D{4}.feature = F.high_p(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_Personal_E_R_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_Personal_E_R_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Personal_E_R_TEST(:,3)) -> best
D{3}.SHAP = DATA_SHAP.SIM.HighDim_Personal_E_SHAP;
D{3}.SHAP_BASE = DATA_SHAP.SIM.HighDim_Personal_E_SHAP_BASE;
D{3}.str = "Sim. (High dim.)";
D{3}.target = "Extraversion";
D{3}.feature = F.high_p(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_Personal_N_R_TEST(:,3))
% mean(DATA_SHAP.SIM.LowDim_Personal_N_R_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Personal_N_R_TEST(:,3)) -> best but negative correlation
D{2}.SHAP = DATA_SHAP.SIM.HighDim_Personal_N_SHAP;
D{2}.SHAP_BASE = DATA_SHAP.SIM.HighDim_Personal_N_SHAP_BASE;
D{2}.str = "Sim. (High dim.)";
D{2}.target = "Neuroticism";
D{2}.feature = F.high_p(:,[3,4]);
% mean(DATA_SHAP.EMP.LowDim_Personal_O_R_TEST(:,3)) -> best
% mean(DATA_SHAP.SIM.LowDim_Personal_O_R_TEST(:,3))
% mean(DATA_SHAP.SIM.HighDim_Personal_O_R_TEST(:,3))
D{1}.SHAP = DATA_SHAP.EMP.LowDim_Personal_O_SHAP;
D{1}.SHAP_BASE = DATA_SHAP.EMP.LowDim_Personal_O_SHAP_BASE;
D{1}.str = "Emp.";
D{1}.target = "Openness";
D{1}.feature = F.low_p(:,[1,2]);

% Blank figure
% ------------
fig = figure(6);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','Figure 6: SHAP values');

dy = [1,-1]*0.18;
tick_y = [];
temp_ax = axes(fig);
temp_ax.Units = 'normalized';
hold(temp_ax,'on');
line(temp_ax,[0,0],[0.3,7.7],'Color',[1,1,1]*0.7,'LineStyle','--','LineWidth',2);
for nTask = 1:numel(D)
    temp_shap = D{nTask}.SHAP;
    temp_shap(:,3) = D{nTask}.SHAP(:,3) + 1; % the first value is 1
    temp_base = D{nTask}.SHAP_BASE;
    temp_feature = D{nTask}.feature;
    SHAP_MEAN = nan(size(temp_feature,1),2);
    PREDICTOR = nan(size(SHAP_MEAN));
    for nSbj = 1:size(SHAP_MEAN,1)
        lgc = temp_shap(:,3) == nSbj;
        SHAP_MEAN(nSbj,:) = mean(temp_shap(lgc,[4,5]),1);
        PREDICTOR(nSbj,:) = D{nTask}.feature(nSbj,:);
    end
    for nAtl = 1:2
        line(temp_ax,[-1,1]*5,[1,1]*(nTask+dy(nAtl)),'Color',[1,1,1]*0.7,'LineStyle',':','LineWidth',1);
        text(temp_ax,-4.8,nTask+dy(nAtl),sprintf('%.02f',mean(abs(SHAP_MEAN(:,nAtl)))),'FontSize',16,'FontWeight','normal','HorizontalAlignment','left','VerticalAlignment','middle');
        [x] = ksj_violin_scatter(SHAP_MEAN(:,nAtl),nTask,21,0.22,0);
        scatter(temp_ax,SHAP_MEAN(:,nAtl),x+dy(nAtl),20,PREDICTOR(:,nAtl),'filled','Marker','o','MarkerFaceAlpha',0.6,'MarkerEdgeColor','none','LineWidth',1);
        colormap(temp_ax,"jet");
        clim(temp_ax,[0.1,0.8]);
        tick_y = [tick_y,nTask+dy(nAtl)*-1];
    end
    annotation(fig,'textbox', [0.15,0.17+(nTask-1)*(0.76/7),0.2,0.01],'string',D{nTask}.str,'edgecolor','none','fontsize',14,'fontweight','n','horizontalalignment','center','VerticalAlignment','middle');
    annotation(fig,'textbox', [0.01,0.17+(nTask-1)*(0.76/7),0.2,0.01],'string',D{nTask}.target,'edgecolor','none','fontsize',18,'fontweight','b','horizontalalignment','left','VerticalAlignment','middle');
end
annotation(fig,'textbox', [0.15,0.9,0.2,0.01],'string','Best predictor','edgecolor','none','fontsize',16,'fontweight','b','horizontalalignment','center','VerticalAlignment','middle');
annotation(fig,'textbox', [0.42,0.88,0.1,0.01],'string','$mean(|$SHAP$|)$','edgecolor','none','fontsize',16,'fontweight','n','horizontalalignment','left','VerticalAlignment','middle','Interpreter','latex');
% annotation(fig,'textbox', [0.30,0.88,0.1,0.01],'string','Atlas','edgecolor','none','fontsize',14,'fontweight','n','horizontalalignment','right','VerticalAlignment','middle');
ylim(temp_ax,[0.3,7.7]);
xlim(temp_ax,[-1,1]*5);
set(temp_ax,'FontSize',14,'FontWeight','normal');
set(temp_ax,'YTick',tick_y,'YTickLabel',{'Harvard-Oxford','Schaefer'},'FontSize',14,'FontWeight','normal');
xlabel(temp_ax,'SHAP values','FontSize',16,'FontWeight','normal');
grid(temp_ax,"off");
temp_ax.Position = [0.42,0.1,0.5,0.8];
hcb = colorbar(temp_ax);
hcb.Position = [0.93,0.1,0.01,0.8];
hcb.Ticks = -1.0:0.1:1.0;
hcb.Limits = [0.099,0.8];
hcb.FontSize = 12;
hcb.Label.String = 'Connectome relationship (predictor)';
hcb.Label.FontSize = 16;

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./figure6.pdf');