clc
clear
close all
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

% Targets
% -------
taskType = ['A','C','E','N','O'];
taskTypeFull = ["Agreeableness","Conscientiousness","Extraversion","Neuroticism","Openness"];
featureType = ["Emp.","Sim. (Low dim.)","Sim. (High dim.)"];
V = [];
V.O = readmatrix('../data/Personal_O_for_Personality_N269_HighDim.csv','Delimiter',',');
V.C = readmatrix('../data/Personal_C_for_Personality_N269_HighDim.csv','Delimiter',',');
V.E = readmatrix('../data/Personal_E_for_Personality_N269_HighDim.csv','Delimiter',',');
V.A = readmatrix('../data/Personal_A_for_Personality_N269_HighDim.csv','Delimiter',',');
V.N = readmatrix('../data/Personal_N_for_Personality_N269_HighDim.csv','Delimiter',',');
V.I = readmatrix('../data/CogTotalComp_Unadj_for_CogTotalComp_Unadj_N268_HighDim.csv','Delimiter',',');
V.S = readmatrix('../data/Female_Male_for_Classification_N270_HighDim.csv','Delimiter',',');

% Coefficients of ML
% ------------------
DATA = [];
test_str = ["Personal_A","Personal_C","Personal_E","Personal_N","Personal_O"];
featureCond = {'EMP','SIM','EMPSIM'};
dimCond = ["LowDim","HighDim"];
for nCond = 1:2 %numel(featureCond)
    for nDimCond = 1:2
        if nCond == 1 && nDimCond == 2
            fprintf('Skip %s, %s\n',featureCond{nCond},dimCond(nDimCond));
        else
            for nTest = 1:5
                temp_r_beta = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_BETA.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                temp_idx    = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                temp_r      = readmatrix(sprintf('../prediction/HCP_N269_%s_NestedCV_CORR_%s_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                eval(sprintf('DATA.%s.%s_%s_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
                eval(sprintf('DATA.%s.%s_%s_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
                eval(sprintf('DATA.%s.%s_%s_R_TEST = temp_r;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
            end

            temp_r_beta = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_BETA.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_idx    = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_r      = readmatrix(sprintf('../prediction/HCP_N268_%s_NestedCV_CORR_COG_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA.%s.%s_COG_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_COG_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_COG_R_TEST = temp_r;',featureCond{nCond},dimCond(nDimCond)));

            temp_r_beta = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_BETA.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_idx    = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_accu   = readmatrix(sprintf('../classification/HCP_N270_%s_NestedCV_%s_ACCURACY_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA.%s.%s_Classification_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_Classification_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA.%s.%s_Classification_ACCU_TEST = temp_accu;',featureCond{nCond},dimCond(nDimCond)));
        end
    end
end

% Permutation test
% ----------------
DATA_PERM = [];
for nCond = 1:2
    for nDimCond = 1:2
        if nCond == 1 && nDimCond == 2
            fprintf('Skip %s, %s\n',featureCond{nCond},dimCond(nDimCond));
        else
            for nTest = 1:5
                temp_r_beta = readmatrix(sprintf('../prediction_permutation_test/HCP_N269_%s_NestedCV_CORR_%s_%s_BETA.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                temp_idx    = readmatrix(sprintf('../prediction_permutation_test/HCP_N269_%s_NestedCV_CORR_%s_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                temp_r      = readmatrix(sprintf('../prediction_permutation_test/HCP_N269_%s_NestedCV_CORR_%s_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),test_str(nTest),featureCond{nCond}),'Delimiter',' ');
                eval(sprintf('DATA_PERM.%s.%s_%s_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
                eval(sprintf('DATA_PERM.%s.%s_%s_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
                eval(sprintf('DATA_PERM.%s.%s_%s_R_TEST = temp_r;',featureCond{nCond},dimCond(nDimCond),test_str(nTest)));
            end

            temp_r_beta = readmatrix(sprintf('../prediction_permutation_test/HCP_N268_%s_NestedCV_CORR_COG_%s_BETA.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_idx    = readmatrix(sprintf('../prediction_permutation_test/HCP_N268_%s_NestedCV_CORR_COG_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_r      = readmatrix(sprintf('../prediction_permutation_test/HCP_N268_%s_NestedCV_CORR_COG_%s_PEARSON_R_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA_PERM.%s.%s_COG_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA_PERM.%s.%s_COG_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA_PERM.%s.%s_COG_R_TEST = temp_r;',featureCond{nCond},dimCond(nDimCond)));

            temp_r_beta = readmatrix(sprintf('../classification_permutation_test/HCP_N270_%s_NestedCV_%s_BETA.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_idx    = readmatrix(sprintf('../classification_permutation_test/HCP_N270_%s_NestedCV_%s_CV_TRAIN_INDEX.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            temp_accu   = readmatrix(sprintf('../classification_permutation_test/HCP_N270_%s_NestedCV_%s_ACCURACY_TEST.txt',dimCond(nDimCond),featureCond{nCond}),'Delimiter',' ');
            eval(sprintf('DATA_PERM.%s.%s_Classification_BETA = temp_r_beta;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA_PERM.%s.%s_Classification_INDEX = temp_idx;',featureCond{nCond},dimCond(nDimCond)));
            eval(sprintf('DATA_PERM.%s.%s_Classification_ACCU_TEST = temp_accu;',featureCond{nCond},dimCond(nDimCond)));
        end
    end
end
%% S-Figure 1
cmap = lines(5);
l = size(F.high_c,1);

% Blank figure
% ------------
fig = figure(11);clf;set(gcf,'Color','w','Position',[1,1,500,500],'Name','S-Figure 1: Linear model');

% Group difference (male vs. female)
% ----------------------------------
y = V.S; % female = 0; male = 1;
lgc_male = V.S == 1;
for nFeature = 1:3
    if nFeature == 1
        X = F.low_c(:,1:2);
    elseif nFeature == 2
        X = F.low_c(:,3:4);
    elseif nFeature == 3
        X = F.high_c(:,3:4);
    end
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    Y = [];
    Y{1} = X(~lgc_male,1); % Schaefer
    Y{2} = X(lgc_male,1);  % Schaefer
    Y{3} = X(~lgc_male,2); % Harvard-Oxford
    Y{4} = X(lgc_male,2);  % Harvard-Oxford
    [p1,~,s1] = ranksum(Y{1},Y{2});
    es1 = s1.zval/sqrt(l);
    [p2,~,s2] = ranksum(Y{3},Y{4});
    es2 = s2.zval/sqrt(l);
    [x] = ksj_violin_scatter(Y{1},1,11,0.60,0);
    hsp1 = scatter(temp_ax,x,Y{1},15,'Marker','o','MarkerFaceColor',cmap(4,:)*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.2);
    [x] = ksj_violin_scatter(Y{2},2,11,0.60,0);
    hsp1 = scatter(temp_ax,x,Y{2},15,'Marker','o','MarkerFaceColor',cmap(5,:)*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.2);
    [x] = ksj_violin_scatter(Y{3},4,11,0.60,0);
    hsp1 = scatter(temp_ax,x,Y{3},15,'Marker','o','MarkerFaceColor',cmap(4,:)*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.2);
    [x] = ksj_violin_scatter(Y{4},5,11,0.60,0);
    hsp1 = scatter(temp_ax,x,Y{4},15,'Marker','o','MarkerFaceColor',cmap(5,:)*0.5,'MarkerEdgeColor','none','MarkerFaceAlpha',0.2);
    hbp1 = boxplot(temp_ax,Y{1},'Positions',1,'Orientation','vertical','Width',0.4,'Colors',cmap(4,:)*0);
    hbp2 = boxplot(temp_ax,Y{2},'Positions',2,'Orientation','vertical','Width',0.4,'Colors',cmap(4,:)*0);
    hbp3 = boxplot(temp_ax,Y{3},'Positions',4,'Orientation','vertical','Width',0.4,'Colors',cmap(4,:)*0);
    hbp4 = boxplot(temp_ax,Y{4},'Positions',5,'Orientation','vertical','Width',0.4,'Colors',cmap(4,:)*0);
    set(hbp1,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
    set(hbp2,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
    set(hbp3,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
    set(hbp4,{'linew'},{1});set(hbp1(6),'Color',[0,0,0]);
    plot(temp_ax,[-0.2,0.2]+1,[1,1]*quantile(Y{1},0.5),'Color','k','LineWidth',3);
    plot(temp_ax,[-0.2,0.2]+2,[1,1]*quantile(Y{2},0.5),'Color','k','LineWidth',3);
    plot(temp_ax,[-0.2,0.2]+4,[1,1]*quantile(Y{3},0.5),'Color','k','LineWidth',3);
    plot(temp_ax,[-0.2,0.2]+5,[1,1]*quantile(Y{4},0.5),'Color','k','LineWidth',3);
    xlim(temp_ax,[0.3,5.7]);
    mm = [min(X(:)),max(X(:))];
    ylim(temp_ax,[mm(1)-(mm(2)-mm(1))*0.1,mm(2)+(mm(2)-mm(1))*0.1]);
    if nFeature == 3
        text(temp_ax,0.55,mm(2)-(mm(2)-mm(1))*0.05,sprintf('es = %.02f (%.02f)',abs(es1),p1),'FontSize',10,'FontWeight','bold');
        text(temp_ax,3.25,mm(1)+(mm(2)-mm(1))*0.05,sprintf('es = %.02f (%.02f)',abs(es2),p2),'FontSize',10,'FontWeight','bold');
    else
        text(temp_ax,0.55,mm(2)-(mm(2)-mm(1))*0.05,sprintf('es = %.02f (%.02f)',abs(es1),p1),'FontSize',10,'FontWeight','normal');
        text(temp_ax,3.25,mm(1)+(mm(2)-mm(1))*0.05,sprintf('es = %.02f (%.02f)',abs(es2),p2),'FontSize',10,'FontWeight','normal');
    end
    grid(temp_ax,'on');
    temp_ax.Position = [0.11+0.07,0.7-(nFeature-1)*0.3,0.36,0.24];
    temp_ax.Box = 'on';
    set(temp_ax,'YTick',-1:0.1:1);
    if nFeature == 1
        set(temp_ax,'XTick',[1,2,4,5],'XTicklabel',"",'FontSize',10,'FontWeight','normal');
        ylabel(temp_ax,'Corr(eFC, eSC)','FontSize',12,'FontWeight','normal');
        title(temp_ax,'Sex difference','FontSize',16,'FontWeight','normal');
    elseif nFeature == 2
        set(temp_ax,'XTick',[1,2,4,5],'XTicklabel',"",'FontSize',10,'FontWeight','normal');
        ylabel(temp_ax,'Corr(eFC, sFC)','FontSize',12,'FontWeight','normal');
    elseif nFeature == 3
        annotation(fig,'textbox', [0.255,0.06,0.01,0.01],'string','Schaefer','edgecolor','none','fontsize',12,'fontweight','normal','horizontalalignment','center');
        annotation(fig,'textbox', [0.45,0.06,0.01,0.01],'string','Harvard-Oxford','edgecolor','none','fontsize',12,'fontweight','normal','horizontalalignment','center');
        set(temp_ax,'XTick',[1,2,4,5],'XTicklabel',["Female","Male","Female","Male"],'FontSize',10,'FontWeight','normal');
        ylabel(temp_ax,'Corr(eFC, sFC)','FontSize',12,'FontWeight','normal');
    end
    text_ax = axes(fig);
    text_ax.Units = 'normalized';
    text_ax.Position = [0.045,0.82-(nFeature-1)*0.3,0.01,0.01];
    axis(text_ax,'off');
    text(text_ax,0.0,0.0,featureType(nFeature),'FontSize',14,'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
    annotation(fig,'line', [0.07,0.07],[0.7,0.94]-(nFeature-1)*0.3, 'color', 'k', 'linewidth', 2);
end

% Linear model for cognition
% --------------------------
y = V.I;
l = size(F.high_i,1);
for nFeature = 1:3
    if nFeature == 1
        X = zscore(F.low_i(:,1:2),0,1);
    elseif nFeature == 2
        X = zscore(F.low_i(:,3:4),0,1);
    elseif nFeature == 3
        X = zscore(F.high_i(:,3:4),0,1);
    end
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    M    = fitlm(X,y,"linear");
    yhat = predict(M,X);
    scatter(temp_ax,y,yhat,20,'Marker','o','MarkerFaceColor',cmap(nFeature,:)*0.7,'MarkerFaceAlpha',0.3,'MarkerEdgeColor','none','MarkerEdgeAlpha',1,'LineWidth',1);
    Y = [ones(l,1),y];
    B = inv(Y'*Y)*Y'*yhat;
    [r,p] = corr(y,yhat,'type','Pearson');
    line(temp_ax,[80,160],B(2)*[80,160] + B(1),'Color',[cmap(nFeature,:)*0.4,0.7],'linewidth',2);
    xlim(temp_ax,[min(y)-(max(y)-min(y))*0.05,max(y)+(max(y)-min(y))*0.05]);
    ylim(temp_ax,[floor(mean(y)-11.0),ceil(mean(y)+8.0)]);
    mm = get(temp_ax,'ylim');
    if nFeature == 2
        text(temp_ax,min(y)+(max(y)-min(y))*0.02,mm(1)+(mm(2)-mm(1))*0.2,sprintf('RMSE = %.02f',M.RMSE),'FontSize',10,'FontWeight','bold');
        text(temp_ax,min(y)+(max(y)-min(y))*0.02,mm(1)+(mm(2)-mm(1))*0.1,sprintf('r = %.02f (%.02f)',r,p),'FontSize',10,'FontWeight','bold');
    else
        text(temp_ax,min(y)+(max(y)-min(y))*0.02,mm(1)+(mm(2)-mm(1))*0.2,sprintf('RMSE = %.02f',M.RMSE),'FontSize',10,'FontWeight','normal');
        text(temp_ax,min(y)+(max(y)-min(y))*0.02,mm(1)+(mm(2)-mm(1))*0.1,sprintf('r = %.02f (%.02f)',r,p),'FontSize',10,'FontWeight','normal');
    end
    set(temp_ax,'XTick',80:20:160,'FontSize',10,'FontWeight','normal');
    grid(temp_ax,'on');
    temp_ax.Position = [0.11+0.57,0.7-(nFeature-1)*0.3,0.24,0.24];
    temp_ax.Box = 'on';
    if nFeature == 1
        title(temp_ax,'Cognition','FontSize',16,'FontWeight','normal');
    elseif nFeature == 3
        xlabel(temp_ax,'Measured score','FontSize',12,'FontWeight','normal');
    end
    ylabel(temp_ax,'Predicted score','FontSize',12,'FontWeight','normal');
end

annotation(fig,'textbox', [0.11,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.11,0.68,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.11,0.38,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.60,0.98,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.60,0.68,0.01,0.01],'string','e','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.60,0.38,0.01,0.01],'string','f','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./suppl_figure1.pdf');
%% S-Figure 2
cmap = lines(5);
l = size(F.high_p,1);

% Blank figure
% ------------
fig = figure(12);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','S-Figure 2: Linear model');

for nTask = 1:numel(taskType)
    eval(sprintf('y = V.%s;',taskType(nTask)));

    % Linear model
    % ------------
    for nFeature = 1:3
        if nFeature == 1
            X = zscore(F.low_p(:,1:2),0,1);
        elseif nFeature == 2
            X = zscore(F.low_p(:,3:4),0,1);
        elseif nFeature == 3
            X = zscore(F.high_p(:,3:4),0,1);
        end
        temp_ax = axes(fig);
        temp_ax.Units = 'normalized';
        hold(temp_ax,'on');
        M    = fitlm(X,y,"linear");
        yhat = predict(M,X);
        scatter(temp_ax,y,yhat,20,'Marker','o','MarkerFaceColor',cmap(nFeature,:)*0.7,'MarkerFaceAlpha',0.3,'MarkerEdgeColor','none','MarkerEdgeAlpha',1,'LineWidth',1);
        Y = [ones(l,1),y];
        B = inv(Y'*Y)*Y'*yhat;
        [r,p] = corr(y,yhat,'type','Pearson');
        line(temp_ax,[0,50],B(2)*[0,50] + B(1),'Color',[cmap(nFeature,:)*0.4,0.7],'linewidth',2);
        xlim(temp_ax,[min(y)-(max(y)-min(y))*0.05,max(y)+(max(y)-min(y))*0.05]);
        ylim(temp_ax,[floor(mean(y)-4.0),ceil(mean(y)+4.0)]);
        mm = get(temp_ax,'ylim');
        if any([nTask == 1 && nFeature == 3;nTask == 2 && nFeature == 3;nTask == 3 && nFeature == 3;nTask == 4 && nFeature == 2;nTask == 5 && nFeature == 1]) && p <0.05
            text(temp_ax,min(y)+(max(y)-min(y))*0.04,mm(1)+(mm(2)-mm(1))*0.2,sprintf('RMSE = %.02f',M.RMSE),'FontSize',10,'FontWeight','bold');
            text(temp_ax,min(y)+(max(y)-min(y))*0.04,mm(1)+(mm(2)-mm(1))*0.1,sprintf('r = %.02f (%.02f)',r,p),'FontSize',10,'FontWeight','bold');
        else
            text(temp_ax,min(y)+(max(y)-min(y))*0.04,mm(1)+(mm(2)-mm(1))*0.2,sprintf('RMSE = %.02f',M.RMSE),'FontSize',10,'FontWeight','normal');
            text(temp_ax,min(y)+(max(y)-min(y))*0.04,mm(1)+(mm(2)-mm(1))*0.1,sprintf('r = %.02f (%.02f)',r,p),'FontSize',10,'FontWeight','normal');
        end
        set(temp_ax,'XTick',0:10:60,'FontSize',10,'FontWeight','normal');
        grid(temp_ax,'on');
        temp_ax.Position = [0.11+(nTask-1)*0.18,0.7-(nFeature-1)*0.3,0.12,0.24];
        temp_ax.Box = 'on';
        if nFeature == 1
            title(temp_ax,taskTypeFull(nTask),'FontSize',16,'FontWeight','normal');
        elseif nFeature == 3
            xlabel(temp_ax,'Measured score','FontSize',12,'FontWeight','normal');
        end
        if nTask == 1
            ylabel(temp_ax,'Predicted score','FontSize',12,'FontWeight','normal');
            text_ax = axes(fig);
            text_ax.Units = 'normalized';
            text_ax.Position = [0.045,0.82-(nFeature-1)*0.3,0.01,0.01];
            axis(text_ax,'off');
            text(text_ax,0.0,0.0,featureType(nFeature),'FontSize',14,'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
            annotation(fig,'line', [0.06,0.06],[0.7,0.94]-(nFeature-1)*0.3, 'color', 'k', 'linewidth', 2);
        end
    end
end
annotation(fig,'textbox', [0.08+0.18*0,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*1,0.98,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*2,0.98,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*3,0.98,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*4,0.98,0.01,0.01],'string','e','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.08+0.18*0,0.68,0.01,0.01],'string','f','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*1,0.68,0.01,0.01],'string','g','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*2,0.68,0.01,0.01],'string','h','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*3,0.68,0.01,0.01],'string','i','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*4,0.68,0.01,0.01],'string','j','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.08+0.18*0,0.38,0.01,0.01],'string','k','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*1,0.38,0.01,0.01],'string','l','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*2,0.38,0.01,0.01],'string','m','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*3,0.38,0.01,0.01],'string','n','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.08+0.18*4,0.38,0.01,0.01],'string','o','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./suppl_figure2.pdf');
%% S-Figure 3

% Blank figure
% ------------
fig = figure(13);clf;set(gcf,'Color','w','Position',[1,1,500,500],'Name','S-Figure 3: BETA');

% Group difference (male vs. female)
% ----------------------------------
y = V.S; % female = 0; male = 1;
lgc_male = V.S == 1;
for nFeature = 1:3
    if nFeature == 1
        B = DATA.EMP.LowDim_Classification_BETA;
        R = DATA.EMP.LowDim_Classification_ACCU_TEST;
        I = DATA.EMP.LowDim_Classification_INDEX;
    elseif nFeature == 2
        B = DATA.SIM.LowDim_Classification_BETA;
        R = DATA.SIM.LowDim_Classification_ACCU_TEST;
        I = DATA.SIM.LowDim_Classification_INDEX;
    elseif nFeature == 3
        B = DATA.SIM.HighDim_Classification_BETA;
        R = DATA.SIM.HighDim_Classification_ACCU_TEST;
        I = DATA.SIM.HighDim_Classification_INDEX;
    end
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    line(temp_ax,[0.0,50.0],[0.0,0.0],'Color',[[1,1,1]*0.2,0.7],'linewidth',1);
    for nAtl = 1:2
        temp_y = B(:,nAtl);
        [x] = ksj_violin_scatter(temp_y,nAtl-0.2,15,0.3,0);
        hsp = scatter(temp_ax,x,temp_y,10,R(:,3),'filled','Marker','o','MarkerFaceAlpha',0.3,'MarkerEdgeColor','none');
        colormap(temp_ax,colors(128:end,:));
        clim(temp_ax,[0.5,0.8]);
        hbp = boxplot(temp_ax,temp_y,'Positions',nAtl+0.2,'Orientation','vertical','Width',0.16,'Colors',[1,1,1]*0.0);
        set(hbp,{'linew'},{1});set(hbp(6),'Color',[0,0,0]);
        set(hbp(7),'marker','none');
        plot(temp_ax,[-0.08,0.08]+nAtl+0.2,[1,1]*quantile(temp_y,0.5),'Color','k','LineWidth',3);
    end
    xlim(temp_ax,[0.3,2.7]);
    ylim(temp_ax,[-1.0,1.0]);
    set(temp_ax,'XTick',1:2,'FontSize',10,'FontWeight','normal');
    grid(temp_ax,'on');
    if nFeature == 1
        set(temp_ax,'XTick',[1,2],'XTickLabel',["",""],'FontSize',12,'FontWeight','normal');
        title(temp_ax,'Sex classification','FontSize',16,'FontWeight','normal');
    elseif nFeature == 3
        xlabel(temp_ax,'Feature','FontSize',12,'FontWeight','normal');
        set(temp_ax,'XTick',[1,2],'XTickLabel',["Sch.","H.O."],'FontSize',12,'FontWeight','normal');
        hcb = colorbar(temp_ax);
        hcb.Position = [0.43,0.1,0.01,0.84];
        hcb.Ticks = 0.5:0.1:0.8;
        hcb.Limits = [0.5,0.8];
        hcb.FontSize = 12;
        hcb.Label.String = 'Balanced accuracy in unseen testing sets';
        hcb.Label.FontSize = 16;
    end
    temp_ax.Position = [0.11+0.07,0.7-(nFeature-1)*0.3,0.24,0.24];
    temp_ax.Box = 'on';
    grid(temp_ax,'on');
    if nFeature == 1
        set(temp_ax,'XTick',[1,2],'XTicklabel',"",'FontSize',12,'FontWeight','normal');
        title(temp_ax,'Sex difference','FontSize',16,'FontWeight','normal');
    elseif nFeature == 2
        set(temp_ax,'XTick',[1,2],'XTicklabel',"",'FontSize',12,'FontWeight','normal');
    elseif nFeature == 3
        set(temp_ax,'XTick',[1,2],'XTickLabel',["Sch.","H.O."],'FontSize',12,'FontWeight','normal');
    end
    ylabel(temp_ax,'Coefficients','FontSize',12,'FontWeight','normal');
    text_ax = axes(fig);
    text_ax.Units = 'normalized';
    text_ax.Position = [0.045,0.82-(nFeature-1)*0.3,0.01,0.01];
    axis(text_ax,'off');
    text(text_ax,0.0,0.0,featureType(nFeature),'FontSize',14,'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
    annotation(fig,'line', [0.07,0.07],[0.7,0.94]-(nFeature-1)*0.3, 'color', 'k', 'linewidth', 2);
end

% Linear model for cognition
% --------------------------
y = V.I;
for nFeature = 1:3
    if nFeature == 1
        B = DATA.EMP.LowDim_COG_BETA;
        R = DATA.EMP.LowDim_COG_R_TEST;
        I = DATA.EMP.LowDim_COG_INDEX;
    elseif nFeature == 2
        B = DATA.SIM.LowDim_COG_BETA;
        R = DATA.SIM.LowDim_COG_R_TEST;
        I = DATA.SIM.LowDim_COG_INDEX;
    elseif nFeature == 3
        B = DATA.SIM.HighDim_COG_BETA;
        R = DATA.SIM.HighDim_COG_R_TEST;
        I = DATA.SIM.HighDim_COG_INDEX;
    end
    temp_ax = axes(fig);
    temp_ax.Units = 'normalized';
    hold(temp_ax,'on');
    line(temp_ax,[0.0,50.0],[0.0,0.0],'Color',[[1,1,1]*0.2,0.7],'linewidth',1);
    for nAtl = 1:2
        temp_y = B(:,nAtl);
        [x] = ksj_violin_scatter(temp_y,nAtl-0.2,15,0.3,0);
        hsp = scatter(temp_ax,x,temp_y,10,R(:,3),'filled','Marker','o','MarkerFaceAlpha',0.3,'MarkerEdgeColor','none');
        colormap(temp_ax,colors);
        clim(temp_ax,[-0.4,0.4]);
        hbp = boxplot(temp_ax,temp_y,'Positions',nAtl+0.2,'Orientation','vertical','Width',0.16,'Colors',[1,1,1]*0.0);
        set(hbp,{'linew'},{1});set(hbp(6),'Color',[0,0,0]);
        set(hbp(7),'marker','none');
        plot(temp_ax,[-0.08,0.08]+nAtl+0.2,[1,1]*quantile(temp_y,0.5),'Color','k','LineWidth',3);
    end
    xlim(temp_ax,[0.3,2.7]);
    ylim(temp_ax,[-2.5,2.5]);
    set(temp_ax,'XTick',1:2,'FontSize',10,'FontWeight','normal');
    grid(temp_ax,'on');
    if nFeature == 1
        set(temp_ax,'XTick',[1,2],'XTickLabel',["",""],'FontSize',12,'FontWeight','normal');
        title(temp_ax,'Cognition','FontSize',16,'FontWeight','normal');
    elseif nFeature == 2
        set(temp_ax,'XTick',[1,2],'XTicklabel',"",'FontSize',12,'FontWeight','normal');
    elseif nFeature == 3
        xlabel(temp_ax,'Feature','FontSize',12,'FontWeight','normal');
        set(temp_ax,'XTick',[1,2],'XTickLabel',["Sch.","H.O."],'FontSize',12,'FontWeight','normal');
        hcb = colorbar(temp_ax);
        hcb.Position = [0.87,0.1,0.01,0.84];
        hcb.Ticks = -0.5:0.1:0.5;
        hcb.Limits = [-0.4,0.4];
        hcb.FontSize = 12;
        hcb.Label.String = 'Pearson''s correlation in unseen testing sets';
        hcb.Label.FontSize = 16;
    end
    grid(temp_ax,'on');
    ylabel(temp_ax,'Coefficients','FontSize',12,'FontWeight','normal');
    temp_ax.Position = [0.11+0.51,0.7-(nFeature-1)*0.3,0.24,0.24];
    temp_ax.Box = 'on';
    ylabel(temp_ax,'Coefficients','FontSize',12,'FontWeight','normal');
end

annotation(fig,'textbox', [0.11,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.11,0.68,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.11,0.38,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.57,0.98,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.57,0.68,0.01,0.01],'string','e','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.57,0.38,0.01,0.01],'string','f','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./suppl_figure3.pdf');
%% S-Figure 4 (BETA)

% Blank figure
% ------------
fig = figure(14);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','S-Figure 4: BETA');

for nTask = 1:numel(taskType)
    eval(sprintf('y = V.%s;',taskType(nTask)));

    % Linear model
    % ------------
    for nFeature = 1:3
        if nFeature == 1
            eval(sprintf('B = DATA.EMP.LowDim_%s_BETA;',test_str(nTask)));
            eval(sprintf('R = DATA.EMP.LowDim_%s_R_TEST;',test_str(nTask)));
            eval(sprintf('I = DATA.EMP.LowDim_%s_INDEX;',test_str(nTask)));
            X = zscore(F.low_p(:,1:2),0,1);
            XO = F.low_p(:,1:2);
        elseif nFeature == 2
            eval(sprintf('B = DATA.SIM.LowDim_%s_BETA;',test_str(nTask)));
            eval(sprintf('R = DATA.SIM.LowDim_%s_R_TEST;',test_str(nTask)));
            eval(sprintf('I = DATA.SIM.LowDim_%s_INDEX;',test_str(nTask)));
            X = zscore(F.low_p(:,3:4),0,1);
            XO = F.low_p(:,3:4);
        elseif nFeature == 3
            eval(sprintf('B = DATA.SIM.HighDim_%s_BETA;',test_str(nTask)));
            eval(sprintf('R = DATA.SIM.HighDim_%s_R_TEST;',test_str(nTask)));
            eval(sprintf('I = DATA.SIM.HighDim_%s_INDEX;',test_str(nTask)));
            X = zscore(F.high_p(:,3:4),0,1);
            XO = F.high_p(:,3:4);
        end
        temp_ax = axes(fig);
        temp_ax.Units = 'normalized';
        hold(temp_ax,'on');
        line(temp_ax,[0.0,50.0],[0.0,0.0],'Color',[[1,1,1]*0.2,0.7],'linewidth',1);
        for nAtl = 1:2
            temp_y = B(:,nAtl);
            [x] = ksj_violin_scatter(temp_y,nAtl-0.2,15,0.3,0);
            hsp = scatter(temp_ax,x,temp_y,10,R(:,3),'filled','Marker','o','MarkerFaceAlpha',0.3,'MarkerEdgeColor','none');
            colormap(temp_ax,colors);
            clim(temp_ax,[-0.4,0.4]);
            hbp = boxplot(temp_ax,temp_y,'Positions',nAtl+0.2,'Orientation','vertical','Width',0.16,'Colors',[1,1,1]*0.0);
            set(hbp,{'linew'},{1});set(hbp(6),'Color',[0,0,0]);
            set(hbp(7),'marker','none');
            plot(temp_ax,[-0.08,0.08]+nAtl+0.2,[1,1]*quantile(temp_y,0.5),'Color','k','LineWidth',3);
        end
        xlim(temp_ax,[0.3,2.7]);
        ylim(temp_ax,[-2.5,2.5]);
        set(temp_ax,'XTick',1:2,'FontSize',10,'FontWeight','normal');
        grid(temp_ax,'on');
        if nFeature == 1
            set(temp_ax,'XTick',[1,2],'XTickLabel',["",""],'FontSize',12,'FontWeight','normal');
            title(temp_ax,taskTypeFull(nTask),'FontSize',16,'FontWeight','normal');
        elseif nFeature == 3
            xlabel(temp_ax,'Feature','FontSize',12,'FontWeight','normal');
            set(temp_ax,'XTick',[1,2],'XTickLabel',["Sch.","H.O."],'FontSize',12,'FontWeight','normal');
            if nTask == 5
                hcb = colorbar(temp_ax);
                hcb.Position = [0.91,0.1,0.01,0.84];
                hcb.Ticks = -0.5:0.1:0.5;
                hcb.Limits = [-0.4,0.4];
                hcb.FontSize = 12;
                hcb.Label.String = 'Pearson''s correlation in unseen testing sets';
                hcb.Label.FontSize = 16;
            end
        end
        if nTask == 1
            ylabel(temp_ax,'Coefficients','FontSize',12,'FontWeight','normal');
            text_ax = axes(fig);
            text_ax.Units = 'normalized';
            text_ax.Position = [0.025,0.82-(nFeature-1)*0.3,0.01,0.01];
            axis(text_ax,'off');
            text(text_ax,0.0,0.0,featureType(nFeature),'FontSize',14,'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
            annotation(fig,'line', [0.04,0.04],[0.7,0.94]-(nFeature-1)*0.3, 'color', 'k', 'linewidth', 2);
        end
        temp_ax.Position = [0.09+(nTask-1)*0.17,0.7-(nFeature-1)*0.3,0.12,0.24];
        temp_ax.Box = 'on';
    end
end
annotation(fig,'textbox', [0.065+0.17*0,0.98,0.01,0.01],'string','a','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*1,0.98,0.01,0.01],'string','b','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*2,0.98,0.01,0.01],'string','c','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*3,0.98,0.01,0.01],'string','d','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*4,0.98,0.01,0.01],'string','e','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.065+0.17*0,0.68,0.01,0.01],'string','f','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*1,0.68,0.01,0.01],'string','g','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*2,0.68,0.01,0.01],'string','h','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*3,0.68,0.01,0.01],'string','i','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*4,0.68,0.01,0.01],'string','j','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

annotation(fig,'textbox', [0.065+0.17*0,0.38,0.01,0.01],'string','k','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*1,0.38,0.01,0.01],'string','l','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*2,0.38,0.01,0.01],'string','m','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*3,0.38,0.01,0.01],'string','n','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
annotation(fig,'textbox', [0.065+0.17*4,0.38,0.01,0.01],'string','o','edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./suppl_figure4.pdf');
%% S-Figure 5, Permutation tests
cmap = lines(5);

% Blank figure
% ------------
fig = figure(15);clf;set(gcf,'Color','w','Position',[1,1,1000,500],'Name','S-Figure 5: Permutation test');

for nTask = 1:7
    for nFeature = 1:3
        temp_ax = axes(fig);
        temp_ax.Units = 'normalized';
        hold(temp_ax,'on');

        % Classification
        % --------------
        if nTask == 1
            if nFeature == 1
                y_perm = DATA_PERM.EMP.LowDim_Classification_ACCU_TEST(:,3);
                y_real = DATA.EMP.LowDim_Classification_ACCU_TEST(:,3);
            elseif nFeature == 2
                y_perm = DATA_PERM.SIM.LowDim_Classification_ACCU_TEST(:,3);
                y_real = DATA.SIM.LowDim_Classification_ACCU_TEST(:,3);
            elseif nFeature == 3
                y_perm = DATA_PERM.SIM.HighDim_Classification_ACCU_TEST(:,3);
                y_real = DATA.SIM.HighDim_Classification_ACCU_TEST(:,3);
            end
            line(temp_ax,[0.5,0.5],[0,1.05],'LineWidth',1,'Color',[0,0,0],'LineStyle','-');
            [p,~,s] = ranksum(y_real,y_perm,'alpha',0.05);
            es = s.zval/sqrt(numel(y_perm));
            [N,EDGES] = histcounts(y_perm,0.00:0.04:1.0);
            x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
            y = N/max(N);
            hb1 = bar(temp_ax,x,y,'FaceColor',[1,1,1]*0.5,'FaceAlpha',0.3,'EdgeColor','none');
            xq = 0.00:0.01/10:1.0;
            vq = interp1(x,y,xq,'pchip');
            temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],[1,1,1]*0.5,'LineWidth',2.0);
            temp_hp.FaceAlpha = 0.2;
            temp_hp.EdgeColor = [1,1,1]*0.5;
            temp_hp.EdgeAlpha = 0.5;
            hb1 = temp_hp;
            [N,EDGES] = histcounts(y_real,0.00:0.04:1.0);
            x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
            y = N/max(N);
            hb1 = bar(temp_ax,x,y,'FaceColor',cmap(nFeature,:),'FaceAlpha',0.3,'EdgeColor','none');
            xq = 0.00:0.01/10:1.0;
            vq = interp1(x,y,xq,'pchip');
            temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(nFeature,:),'LineWidth',2.0);
            temp_hp.FaceAlpha = 0.2;
            temp_hp.EdgeColor = cmap(nFeature,:);
            temp_hp.EdgeAlpha = 0.5;
            hb1 = temp_hp;
            temp_ax.Position = [0.07+(nTask-1)*0.17,0.72-(nFeature-1)*0.3,0.12,0.20];
            temp_ax.Box = 'on';
            grid(temp_ax,'on');
            xlim(temp_ax,[0.35,0.85]);
            set(temp_ax,'XTick',0.4:0.1:0.8,'XTickLabel',["0.4","0.5","0.6","0.7","0.8"],'YTick',0.0:0.5:1.0,'YTickLabel','');
            ylim(temp_ax,[0,1.3]);
            mm = get(temp_ax,'XLim');
            if nFeature == 3
                text(temp_ax,mm(1)+(mm(2)-mm(1))*0.06,1.17,sprintf('es = %.02f (%.02f)',es,p),'FontSize',10,'FontWeight','bold');
            else
                text(temp_ax,mm(1)+(mm(2)-mm(1))*0.06,1.17,sprintf('es = %.02f (%.02f)',es,p),'FontSize',10,'FontWeight','normal');
            end
        
        % Cognition
        % ----------
        elseif nTask >= 2
            if nTask == 2
                if nFeature == 1
                    y_perm = DATA_PERM.EMP.LowDim_COG_R_TEST(:,3);
                    y_real = DATA.EMP.LowDim_COG_R_TEST(:,3);
                elseif nFeature == 2
                    y_perm = DATA_PERM.SIM.LowDim_COG_R_TEST(:,3);
                    y_real = DATA.SIM.LowDim_COG_R_TEST(:,3);
                elseif nFeature == 3
                    y_perm = DATA_PERM.SIM.HighDim_COG_R_TEST(:,3);
                    y_real = DATA.SIM.HighDim_COG_R_TEST(:,3);
                end
            else
                if nFeature == 1
                    eval(sprintf('y_perm = DATA_PERM.EMP.LowDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                    eval(sprintf('y_real = DATA.EMP.LowDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                elseif nFeature == 2
                    eval(sprintf('y_perm = DATA_PERM.SIM.LowDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                    eval(sprintf('y_real = DATA.SIM.LowDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                elseif nFeature == 3
                    eval(sprintf('y_perm = DATA_PERM.SIM.HighDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                    eval(sprintf('y_real = DATA.SIM.HighDim_%s_R_TEST(:,3);',test_str(nTask-2)));
                end
            end
            line(temp_ax,[0,0],[0,1.05],'LineWidth',1,'Color',[0,0,0],'LineStyle','-');
            [p,~,s] = ranksum(y_real,y_perm,'alpha',0.05);
            es = s.zval/sqrt(numel(y_perm));
            [N,EDGES] = histcounts(y_perm,-1.0:0.08:1.0);
            x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
            y = N/max(N);
            hb1 = bar(temp_ax,x,y,'FaceColor',[1,1,1]*0.5,'FaceAlpha',0.3,'EdgeColor','none');
            xq = -1.0:0.01/10:1.0;
            vq = interp1(x,y,xq,'pchip');
            temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],[1,1,1]*0.5,'LineWidth',2.0);
            temp_hp.FaceAlpha = 0.2;
            temp_hp.EdgeColor = [1,1,1]*0.5;
            temp_hp.EdgeAlpha = 0.5;
            hb1 = temp_hp;
            [N,EDGES] = histcounts(y_real,-1.0:0.08:1.0);
            x = EDGES(1:end-1) + (EDGES(2) - EDGES(1))/2;
            y = N/max(N);
            hb1 = bar(temp_ax,x,y,'FaceColor',cmap(nFeature,:),'FaceAlpha',0.3,'EdgeColor','none');
            xq = -1.0:0.01/10:1.0;
            vq = interp1(x,y,xq,'pchip');
            temp_hp = patch(temp_ax,[xq,xq(1)],[vq,0],cmap(nFeature,:),'LineWidth',2.0);
            temp_hp.FaceAlpha = 0.2;
            temp_hp.EdgeColor = cmap(nFeature,:);
            temp_hp.EdgeAlpha = 0.5;
            hb1 = temp_hp;
            temp_ax.Position = [0.07+(nTask-1)*0.13,0.72-(nFeature-1)*0.3,0.12,0.20];
            temp_ax.Box = 'on';
            grid(temp_ax,'on');
            xlim(temp_ax,[-0.5,0.5]);
            set(temp_ax,'XTick',-0.4:0.2:0.4,'YTick',0.0:0.5:1.0,'YTickLabel','','XTickLabelRotation',0);
            ylim(temp_ax,[0,1.3]);
            mm = get(temp_ax,'XLim');
            if any([nTask == 2 && nFeature == 2;nTask == 3 && nFeature == 3;nTask == 4 && nFeature == 3;nTask == 5 && nFeature == 3;nTask == 6 && nFeature == 3;nTask == 7 && nFeature == 1]) && es > 0.0
                text(temp_ax,mm(1)+(mm(2)-mm(1))*0.06,1.17,sprintf('es = %.02f (%.02f)',es,p),'FontSize',10,'FontWeight','bold');
            else
                text(temp_ax,mm(1)+(mm(2)-mm(1))*0.06,1.17,sprintf('es = %.02f (%.02f)',es,p),'FontSize',10,'FontWeight','normal');
            end
        end
        if nTask == 1
            ylabel(temp_ax,'Frequency','FontSize',12,'FontWeight','normal');
            text_ax = axes(fig);
            text_ax.Units = 'normalized';
            text_ax.Position = [0.025,0.82-(nFeature-1)*0.3,0.01,0.01];
            axis(text_ax,'off');
            text(text_ax,0.0,0.0,featureType(nFeature),'FontSize',14,'FontWeight','normal','HorizontalAlignment','center','VerticalAlignment','middle','Rotation',90);
            annotation(fig,'line', [0.04,0.04],[0.72,0.92]-(nFeature-1)*0.3, 'color', 'k', 'linewidth', 2);
        end
        if nFeature == 3
            if nTask == 1
                xlabel(temp_ax,'Balanced accuracy','FontSize',12,'FontWeight','normal');
            else
                xlabel(temp_ax,'Correlation','FontSize',12,'FontWeight','normal');
            end
        elseif nFeature == 1
            if nTask == 1
                title(temp_ax,'Sex classification','FontSize',12,'FontWeight','normal');
            elseif nTask == 2
                title(temp_ax,'Cognition','FontSize',12,'FontWeight','normal');
            else
                title(temp_ax,taskTypeFull(nTask-2),'FontSize',12,'FontWeight','normal');
            end
        end
    end
end
idx_str = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"];
nPlot = 0;
for nFeature = 1:3
    for nTask = 1:7
        nPlot = nPlot + 1;
        annotation(fig,'textbox', [0.063+0.13*(nTask-1),0.97-0.3*(nFeature-1),0.01,0.01],'string',idx_str(nPlot),'edgecolor','none','fontsize',20,'fontweight','b','horizontalalignment','center');
    end
end

% Save
set(fig,'Resize','on','PaperPositionMode','auto','PaperUnits','points','PaperSize',fig.Position([3,4]) + 1);drawnow;
% saveas(fig,'./suppl_figure5.pdf');
