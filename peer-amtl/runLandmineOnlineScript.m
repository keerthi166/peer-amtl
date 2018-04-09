clear;
rng('default');

% Read Landmine data
dataset='landmine';
load('data/landmine.mat')
sTids=[1:10,16:24];
X=X(sTids);
Y=Y(sTids);
N=cellfun(@(x) size(x,1),X);
K= length(Y);

% Add intercept
X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
% Change the labelspace from {0,1} to {-1,1}
Y=cellfun(@(y) 2*y-1,Y,'UniformOutput',false);

Nrun=30;
% CV Settings
kFold = 5; % 5 fold cross validation


%  Parameter: method
% 'PERCEPTRON' - simple perceptron,
% 'PAONE' - Passive Aggressive updates,
% 'PA' - Passive Aggressive updates,
% 'DEKEL' Shared Loss based update for Online MTL,
% 'CAVALLANTI' - Fixed prior multitask updates
% 'OMTRL' - Online Multitask Relationship Learning
% 'SMTL1' - Online Smoothed Multitask Learning with exact update
% 'SMTL2' - Online Smoothed Multitask Learning with Incremental/Exp weight update


trainSize=160;
methods={'RAND','IND','PEERsum','PEERone','PEERmax'};

opts.dataset=dataset;
opts.scoreType='accuracy'; % Choose one: 'accuracy', 'fmeasure'
opts.isHigherBetter=true;
opts.debugMode=false;
opts.verbose=true;
opts.cv=false;
opts.cvFold=kFold;
opts.tepoch=1;

fprintf('Train Size %d\n',trainSize);



N=cellfun(@(x) size(x,1),X);
for kk=1:length(Y)
    if (N(kk)<trainSize)
        X{kk}=[];
        Y{kk}=[];
    end
end
X=X(~cellfun('isempty',X));
Y=Y(~cellfun('isempty',Y));

K= length(Y);
N=cellfun(@(x) size(x,1),X);

fprintf('Selected %d Tasks \n',K);
% Create Train-Test Splits
splits=cell(1,Nrun);
shufIdx=zeros(trainSize*K,Nrun);
for rId=1:Nrun
    splits{rId}=cellfun(@(y,n) cvpartition(y,'HoldOut',n-trainSize),Y,num2cell(N),'UniformOutput',false);
    shufIdx(:,rId)=randperm(trainSize*K)';
end
results=cell(1,length(methods));
lambda_range=[0.1,0.5,1,2,5,10,25,50];

for rId=1:Nrun
    if opts.verbose
        fprintf('Run %d (',rId);
    end
    
    split=splits{rId};
    Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
    Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y, split);
    Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
    Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y, split);
    
    Xt=[];
    Yt=[];
    taskId=[];
    for kk=1:K
        Xt=[Xt;Xtrain{kk}];
        Yt=[Yt;Ytrain{kk}];
        taskId=[taskId;ones(size(Xtrain{kk},1),1)*kk];
    end
    %shufIdx=randperm(size(Xt,1)); % shuffle so task order is random
    Xt=Xt(shufIdx(:,rId),:);
    Yt=Yt(shufIdx(:,rId));
    taskId=taskId(shufIdx(:,rId));
    
    
    
    % Online Training of the model
    if opts.verbose
        fprintf('Exp[');
    end
    for m=1:length(methods)
        method=methods{m};
        opts.method=method;
        b=1;
        bq=1;
        lambda=5;
        switch (method)
            case 'RAND'
                opts.b=b;
                [model,result_run]=peer(Xt,Yt,taskId,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'IND'
                opts.b=b;
                [model,result_run]=peer(Xt,Yt,taskId,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'PEERsum'
                opts.lambda=lambda;
                opts.b=b;
                opts.bq=bq;
                [model,result_run]=peer(Xt,Yt,taskId,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'PEERone'
                opts.lambda=lambda;
                opts.b=b;
                opts.bq=bq;
                [model,result_run]=peer(Xt,Yt,taskId,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'PEERmax'
                opts.lambda=lambda;
                opts.b=b;
                opts.bq=bq;
                [model,result_run]=peer(Xt,Yt,taskId,opts);
                if opts.verbose
                    fprintf('*');
                end
        end
        
        results{m}.models{rId}=model;
        
        results{m}.mistakes(rId)=result_run.cumMistakes(end);
        results{m}.nquery(rId)=result_run.nquery;
        
        results{m}.cum_mistake(rId,:)=result_run.cumMistakes;
        results{m}.cumQueryInterval(rId,:)=result_run.cumQueryInterval;
        results{m}.cumPeerInterval(rId,:)=result_run.cumPeerInterval;
        results{m}.cumQueryPercent(rId,:)=result_run.cumQueryPercent;
        results{m}.timetaken(rId)=(result_run.timetaken);
        
        % Compute Score for testset [score,task_score] = eval_MTL (Y, X, W, probY,scoreType)
        [results{m}.acc_result(rId),results{m}.acc_result_task(rId,:)]=eval_MTL(Ytest, Xtest, model.W,[], 'accuracy');
        [results{m}.fm_result(rId),results{m}.fm_result_task(rId,:)]=eval_MTL(Ytest, Xtest, model.W,[], 'fmeasure');
    end
    if opts.verbose
        fprintf(']:DONE)\n');
    end
end

clear X Y Xtrain Ytrain Xtest Ytest

if opts.verbose
    fprintf('Results: \n');
end
for m=1:length(methods)
    method=methods{m};
    results{m}.meanACC=mean(results{m}.acc_result);
    results{m}.stdACC=std(results{m}.acc_result);
    results{m}.meanFM=mean(results{m}.fm_result);
    results{m}.stdFM=std(results{m}.fm_result);
    
    results{m}.meanMistakes=mean(results{m}.mistakes);
    results{m}.meanNQuery=mean(results{m}.nquery);
    results{m}.meanCumMistakes=mean(results{m}.cum_mistake);
    results{m}.meanCumQueryInterval=mean(results{m}.cumQueryInterval);
    results{m}.meanCumPeerInterval=mean(results{m}.cumPeerInterval);
    results{m}.meanCumQueryPercent=mean(results{m}.cumQueryPercent);
    results{m}.meanTotalTime=mean(results{m}.timetaken);
    
    results{m}.stdMistakes=std(results{m}.mistakes);
    results{m}.stdNQuery=std(results{m}.nquery);
    results{m}.stdCumMistakes=std(results{m}.cum_mistake);
    results{m}.stdCumQueryInterval=std(results{m}.cumQueryInterval);
    results{m}.stdCumPeerInterval=std(results{m}.cumPeerInterval);
    results{m}.stdCumQueryPercent=std(results{m}.cumQueryPercent);
    results{m}.stdTotalTime=std(results{m}.timetaken);
    
    
    results{m}.meanTaskACC=mean(results{m}.acc_result_task);
    results{m}.stdTaskACC=std(results{m}.acc_result_task);
    results{m}.meanTaskFM=mean(results{m}.fm_result_task);
    results{m}.stdTaskFM=std(results{m}.fm_result_task);
    if opts.verbose
        fprintf('Method: %s, Mistake: %f, Nquery:%f, timetaken:%f \n', method,results{m}.meanMistakes,results{m}.meanNQuery,results{m}.meanTotalTime);
    end
end

%save(sprintf('%s-online.mat',dataset));

if opts.verbose    
    fprintf('Accuracy: \n');
    
    for m=1:length(methods)
        method=methods{m};
        fprintf('Method: %s, meanACC: %f, stdACC:%f \n', method,results{m}.meanACC,results{m}.stdACC);
        
    end
    fprintf('F-measure: \n');
    
    for m=1:length(methods)
        method=methods{m};
        fprintf('Method: %s, meanFM: %f, stdFM:%f \n', method,results{m}.meanFM,results{m}.stdFM);
        
    end
    
end
