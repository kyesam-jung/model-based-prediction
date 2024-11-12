function [X,h] = ksj_violin_scatter(vec,pos,bin,wide,fig)
%KSJ_VIOLIN_SCATTER Rearrange x-axis for scatter plot shaped in a violin plot
%
%   [X,h] = KSJ_VIOLIN_SCATTER(vec,pos,bin,wide,fig)
%
%   Inputs:   vec       - Vectors for scatter e.g., [values x 1]
%             pos       - Middle line of the distribution x-axis    (Default = 0)
%             bin       - Number of bins for a violin shape         (Default = 100)
%             wide      - Horizontal width of a violin shape        (Default = 1)
%             fig       - Plot scatter in a given figure handle     (Default = 0)
%
%   Output:   X         - Rearranged x values with respect to 'vec' input
%             h         - Figure handle
%
% See also HISTCOUNTS, SCATTER.
%
% Kyesam Jung, 23 August 2020
if ~exist('pos','var') || isempty(pos)
    pos = 0;
end
if ~exist('bin','var') || isempty(bin)
    bin = 100;
end
if ~exist('wide','var') || isempty(wide)
    wide = 1;
end
if ~exist('fig','var') || isempty(fig)
    fig = 0;
end
[N,EDGES,BIN] = histcounts(vec,bin);
X = nan(numel(vec),1);
wRatio = wide/max(N);
for n = 1:numel(N)
    if N(n) > 1
        x = (0:(N(n)-1)) * wRatio;
        x = x - wRatio*N(n)/2 + pos;
        X(BIN == n) = x;
    elseif N(n) == 1
        x = pos;
        X(BIN == n) = x;
    end 
end
if fig~=0
    figure(fig);
    h = scatter(X,vec);
end
return