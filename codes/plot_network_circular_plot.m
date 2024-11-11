function ax = plot_network_circular_plot(M,ax,thr,MinMax,cmap,L,S,fsz)
%PLOT_NETWORK_CIRCULAR_PLOT(M,fig,thr,MinMax,cmap,L,S)
%
% M      - Connectome
% fig    - Figure handle
% thr    - threshold to draw
% MinMax - Range of colormap
% cmap   - Manual colormap
% L      - Label strings in cells
% S      - Symmetric labels in hemisphere? then 'symmetric'
% fsz    - FontSize
%
% Kyesam Jung, 14 June 2023

N = size(M,1);
if nargin < 2
    ax = axes(gcf);
end
if nargin < 3
    thr = 0.5;
end
if nargin < 4
    MinMax = [min(M(:)),max(M(:))];
end
if nargin < 5
    cmap = jet(1001);
end
if nargin < 6
    L = cell(N,1);
    for n = 1:N
        L{n,1} = num2str(n);
    end
end
if nargin < 7
    S = 'symmetric';
end
if nargin < 8
    fsz = 8;
end
l_cmap = size(cmap,1);
r = 1;
if rem(N,2) == 0 && strcmp(S,'symmetric')
    theta = linspace(pi/2,2.5*pi,N+3)'; theta([1,N/2+2,end]) = [];
    theta((N/2+1):end) = theta(end:-1:(N/2+1));
else
    theta = linspace(pi/2,2.5*pi,N+2)'; theta([1,end]) = [];
end
xy = r .* [cos(theta) sin(theta)];
line(ax,xy(:,1),xy(:,2),'LineStyle','none','Marker','.','MarkerSize',5,'Color','k');hold(ax,'on');
if fsz > 0
    if rem(N,2) == 0 && strcmp(S,'symmetric')
        h1 = text(ax,xy(1:N/2,1).*1.10,xy(1:N/2,2).*1.10,L(1:N/2),'FontSize',fsz,'HorizontalAlignment','center');
        set(h1,{'Rotation'},num2cell(theta(1:N/2)*180/pi+180))
        h2 = text(ax,xy((N/2+1):end,1).*1.10,xy((N/2+1):end,2).*1.10,L((N/2+1):end),'FontSize',fsz,'HorizontalAlignment','center');
        set(h2,{'Rotation'},num2cell(theta((N/2+1):end)*180/pi))
    else
        h = text(ax,xy(:,1).*1.10,xy(:,2).*1.10,L,'FontSize',fsz,'HorizontalAlignment','center');
        set(h,{'Rotation'},num2cell(theta*180/pi))
    end
end
for nr = 1:N
    for nc = 1:N
        if nr < nc && abs(M(nr,nc)) > thr
            c = round(l_cmap/(MinMax(2)-MinMax(1))*(M(nr,nc)-MinMax(1)));
            if c < 1
                c = 1;
            elseif c > l_cmap
                c = l_cmap;
            end
            
            % Bezier curves
            % -------------
            % Q = Bezier([xy(nr,1),0,xy(nc,1); xy(nr,2),0,xy(nc,2)]);
            % Q = Bezier([xy(nr,1),xy(nr,1)*3/4,xy(nc,1)*3/4,xy(nc,1); xy(nr,2),xy(nr,2)*3/4,xy(nc,2)*3/4,xy(nc,2)]);
            Q = Bezier([xy(nr,1),(xy(nr,1)+xy(nc,1))/3,xy(nc,1); xy(nr,2),(xy(nr,2)+xy(nc,2))/3,xy(nc,2)]);
            plot(ax,Q(1,:),Q(2,:),'LineWidth',3*c/size(cmap,1),'Color',[cmap(c,:),0.9]);
            
            % Straight lines
            % --------------
            % line([xy(nr,1),xy(nc,1)],[xy(nr,2),xy(nc,2)],'LineWidth',0.5,'Color',cmap(c,:));
        end
    end
end
% text(ax,0-0.8,1,'Left','HorizontalAlignment','right','FontSize',fsz,'FontWeight','Normal');
% text(ax,0+0.8,1,'Right','HorizontalAlignment','left','FontSize',fsz,'FontWeight','Normal');
return;

% ============================== Bezier curve =============================
%            https://github.com/NumericalMax/Bezier-Curves.git
% =========================================================================
%Approximation einer Bezierkurve (Kurve und ein Konstruktionsschritt
%wird geplottet)
%Eingabeparameter: Kontrollpunkte (2-Dimensionales Array),
%Streckenteilungsparamter t
%Ausgabe: Bezierkurve fuer das gesetzte Kontrollpolygon
%Ersteller: Maximilian Kapsecker
%Version: 1.0
%Example Function Call:
%S = [0,1,2,3,4; 0,1,2,2,1];
%Bezier(S,0.5);
function Q = Bezier(S)
    for k = 0:0.01:1
        Q(:,round(k*100+1)) = deCasteljau(S,k,0);
    end
return;

%De-Casteljau Algorithmus zum Errechnen und Plotten von Bezierkurven
%Eingabeparameter: Kontrollpunkte (2-Dimensionales Array),
%Streckenteilungsparamter t, Boolean Wert ob Kontrollpunkte und 
%Konstruktionspunkte gezeichnet werden sollen
%Ausgabe: Punkt auf der Bezierkurve fuer das gesetzte Kontrollpolygon in
%Abhaengigkeit von t
%Ersteller: Maximilian Kapsecker
%Version: 1.2
function Casteljau = deCasteljau(P_Start,t,draw)
    L = size(P_Start,2);
    %Initialisierung
    %Der erste Eintrag entspricht hierbei 1 fuer den x-Achsen Abschnitt
    %und 2 fuer den y-Achsen Abschnitt
    %Der zweite Eintrag entspricht demnach einem (x,y) Punkt
    %Der dritte Eintrag wird genutzt um anzugeben in welchem Iterationsschritt
    %man sich gerade befindet
    P(:,:,1) = P_Start;
    if draw == 1
        %Minima und Maxima fuer Axen-Skalierung ermitteln
        M1 = max(P(1,:,1));
        M2 = max(P(2,:,1));
        m1 = min(P(1,:,1));
        m2 = min(P(2,:,1));

        %Verschiebung der labels,
        %so dass diese nicht direkt auf den Punkten erscheinen
        dx = 0.1;
        dy = 0.1;

        %Plotten des Kontrollpolygons und skalieren der Axen
        plot(P(1,:,1),P(2,:,1), 'k');
        axis([(m1-1) (M1+1) (m2-1) (M2+1)]);
        title(sprintf('t = %.2f', t));
        hold on;

        %Plotten der Kontrollpunkte
        scatter(P(1,:,1),P(2,:,1),'filled','b');
        for i=1:1:L
            d2 = num2str(i);
            c = ['P(0,' d2 ')'];
            q = text(P(1,i,1)+dx, P(2,i,1)+dy, c);
            set(q, 'FontSize', 10);
        end
        hold on;
    end

    %Konstruktion des Bezierkurvenpunktes
    for i=1:1:L
        for j=1:1:L-i

            %i-ter Iterationsschritt fuer die x-Achse
            P(1,j,i+1) = (1-t)*P(1,j,i) + t*P(1,j+1,i);
            P(2,j,i+1) = (1-t)*P(2,j,i) + t*P(2,j+1,i);

            if draw == 1
                %Plotten der Verbindungsgeraden im i-ten Iterationsschritt
                if j > 1
                    plot(P(1,j-1:j,i+1),P(2,j-1:j,i+1), 'k');
                    hold on;
                end

                %Plotten der Konstruktionspunkte im i-ten Iterationsschritt
                scatter(P(1,j,i+1),P(2,j,i+1),[],i,'filled');
                d1 = num2str(i);
                d2 = num2str(j);
                c = ['P(' d1 ',' d2 ')'];
                q = text(P(1,j,i+1)+dx, P(2,j,i+1)+dy, c);
                set(q, 'FontSize', 10);
                hold on;
            end
        end
    end
    Casteljau = P(:,1,L);
return;