
function [model, results]=peer (X,Y,taskId,opts)
% peer selective sampling for multitask perceptron algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    Vector of labels from all the tasks.
%        X:    matrix of size NxP obtained by concating the data from all
%              the K tasks. N-total number of observations and P-dimension
%              of the feature space
%   taskId:    vector of mapping from observation to task
%   opts:    a struct containing C, tau, rho, sigma, t_tick;
%
% Output:
%   model:  a struct containing SV (the set of idexes for all the support vectors) and alpha (corresponding weights)
%  result:  a struct containing the statistics for the current run
%--------------------------------------------------------------------------

K=length(unique(taskId));
[N,Pd]=size(X);
err_count = 0;
query_count=0;
peer_count=0;

method=opts.method;

b=opts.b;
if strncmp(method,'PEER',4)
    bq=opts.bq;
    lambda=opts.lambda;
end
W=zeros(Pd,K);
tau=ones(K,K)/(K-1) - eye(K,K).*((K-2)/(K-1));


cumMistakes = [];
cumMistakeInterval = [];
cumQueryInterval = [];
cumPeerInterval = [];
ttime=[];
intMatrix=zeros(K);

tepoch=opts.tepoch;
r=zeros(1,K-1);
tic
for t = 1:N,
    
    tX=X(t,:)./norm(X(t,:)); %x_t
    tId=taskId(t); %i_t
    
    
    f_t=tX*W;
    
    %% count the number of errors
    y_hat = sign(f_t(tId));
    if (y_hat==0),
        y_hat=1;
    end
    Q=0;
    switch (method)
        case 'RAND'
            P=binornd(1,rand);
            Z=P;
        case 'IND'
            p=b/(b+abs(f_t(tId)));
            P=binornd(1,p);
            Z=P;
        case 'PEERsum'
            p=b/(b+abs(f_t(tId)));
            P=binornd(1,p);
            Q=0;
            if P==1
                
                
                selTasks=[1:tId-1,tId+1:K];
                fpeer=f_t(selTasks)*tau(tId,selTasks)';
                p=bq/(bq+abs(fpeer));
                Q=binornd(1,p);
                if Q==0
                    peer_count=peer_count+1;
                    y_hat=sign(fpeer);
                end
            end
            Z=P*Q;
        case 'PEERone'
            p=b/(b+abs(f_t(tId)));
            P=binornd(1,p);
            Q=0;
            if P==1
                
                a=1:K;
                selTasks=a(boolean(mnrnd(1,tau(tId,[1:tId-1,tId+1:K]))));
                
                %[~,selTasks]=max(tau(tId,[1:tId-1,tId+1:K]));
                fpeer=f_t(selTasks);
                p=bq/(bq+abs(fpeer));
                Q=binornd(1,p);
                if Q==0
                    peer_count=peer_count+1;
                    intMatrix(tId,selTasks)=intMatrix(tId,selTasks)+1;
                    y_hat=sign(fpeer);
                end
            end
            Z=P*Q;
        case 'PEERmax'
            p=b/(b+abs(f_t(tId)));
            P=binornd(1,p);
            Q=0;
            if P==1
                
                [~,selTasks]=max(f_t([1:tId-1,tId+1:K]));
                fpeer=f_t(selTasks);
                p=bq/(bq+abs(fpeer));
                Q=binornd(1,p);
                if Q==0
                    peer_count=peer_count+1;
                    y_hat=sign(fpeer);
                end
            end
            
            Z=P*Q;
    end

    tY=Y(t); % query its label
   if Z==1
        query_count = query_count + 1;
   end

    if(y_hat~=tY) %&& query_count<0.3*N
        err_count = err_count + 1;
        %query_count = query_count + Z;
        W(:,tId)=W(:,tId) + Z*tX'*tY;
        if strncmp(method,'PEER',4)
            % update tau
            sId=[1:tId-1,tId+1:K];
            l_t=max(0,1-tY.*f_t(sId));
            lambda=sum(l_t);
            if lambda==0
                lambda=1;
            end
            tau_h=tau(tId,sId).*exp(-Z*l_t/lambda);
            tau(tId,sId)=tau_h/sum(tau_h);
        end
    end
    
    if strncmp(method,'PEER',4) && P==1 && Q==0
        W(:,tId)=W(:,tId) + tX'*y_hat;
        %{
        if strncmp(method,'PEER',4)
           % update tau
           sId=[1:tId-1,tId+1:K];
           l_t=max(0,1-y_hat.*f_t(sId));
           tau_h=tau(tId,sId).*exp(-Z*l_t/lambda);
           tau(tId,sId)=tau_h/sum(tau_h);
        end
        %}
    end
    %}
     cumPeerInterval = [cumPeerInterval peer_count];
    cumQueryInterval = [cumQueryInterval query_count];
    %% record performance
    run_time = toc;
    if (mod(t,tepoch)==0),
        cumMistakes = [cumMistakes err_count/t];
        %SVs = [SVs length(SV)]; % Total support vector added since the last epoch to collect stats
        ttime=[ttime run_time]; % Time taken for each observation
    end
end


% return model.W, model.tau
model.W=W;
model.tau=tau;
% Generate peformance stat



results.mistakes=err_count;
results.nquery=query_count;
results.npeer=peer_count;
results.cumMistakes=cumMistakes;
results.cumMistakeInterval=cumMistakeInterval;
results.cumQueryInterval=cumQueryInterval;
results.cumPeerInterval=cumPeerInterval;
results.cumQueryPercent=cumMistakes(round(query_count*[0.1:0.1:1]));
%results.cumPeerPercent=cumMistakes(round(peer_count*[0.1:0.1:1]));
results.timetaken = toc; % training time
results.intMatrix=intMatrix;

end








