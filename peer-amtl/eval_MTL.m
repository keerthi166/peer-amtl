function [score,task_score] = eval_MTL (Y, X, W, probY,scoreType)
% probY: cell array of size task_num, probability or confidence values for predictions if available,
% empty if not
task_num = length(Y);
score=0;
task_score=zeros(task_num,1);
if strcmp(scoreType,'accuracy')
    if isempty(probY)
        probY=cellfun(@(x,t) x*W(:,t), X,num2cell(1:task_num),'UniformOutput',false);
    end
    
    ct=0;
    for t = 1: task_num
        N = size(X{t},1);
        if(length(unique(Y{t}))==1)
            task_score(t)= 0;
            continue;
        end
        corr=sum(sign(probY{t})==Y{t});
        task_score(t)= corr/N;
        score=score+corr/N;
        ct=ct+1;
    end
    score=score/ct;
elseif strcmp(scoreType,'fmeasure')
    if isempty(probY)
        probY=cellfun(@(x,t) x*W(:,t), X,num2cell(1:task_num),'UniformOutput',false);
    end
    
    ct=0;
    for t = 1: task_num
        N = size(X{t},1);
        if(length(unique(Y{t}))==1)
            task_score(t)= 0;
            continue;
        end
        stats=confusionmatStats(Y{t},sign(probY{t}));
        task_score(t)= sum(stats.Fscore)/2;
        score=score+sum(stats.Fscore)/2;
        ct=ct+1;
    end
    score=score/ct;
else
    disp('Unknown error type. Please use one of the valid options.\n');
end


    function stats = confusionmatStats(group,grouphat)
        % INPUT
        % group = true class labels
        % grouphat = predicted class labels
        %
        % OR INPUT
        % stats = confusionmatStats(group);
        % group = confusion matrix from matlab function (confusionmat)
        %
        % OUTPUT
        % stats is a structure array
        % stats.confusionMat
        %               Predicted Classes
        %                    p'    n'
        %              ___|_____|_____|
        %       Actual  p |     |     |
        %      Classes  n |     |     |
        %
        % stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
        % stats.precision = TP / (TP + FP)                  % for each class label
        % stats.sensitivity = TP / (TP + FN)                % for each class label
        % stats.specificity = TN / (FP + TN)                % for each class label
        % stats.recall = sensitivity                        % for each class label
        % stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
        %
        % TP: true positive, TN: true negative,
        % FP: false positive, FN: false negative
        %
        
        field1 = 'confusionMat';
        if nargin < 2
            value1 = group;
        else
            [value1,gorder] = confusionmat(group,grouphat);
        end
        
        numOfClasses = size(value1,1);
        totalSamples = sum(sum(value1));
        
        [TP,TN,FP,FN,accuracy,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
        for class = 1:numOfClasses
            TP(class) = value1(class,class);
            tempMat = value1;
            tempMat(:,class) = []; % remove column
            tempMat(class,:) = []; % remove row
            TN(class) = sum(sum(tempMat));
            FP(class) = sum(value1(:,class))-TP(class);
            FN(class) = sum(value1(class,:))-TP(class);
        end
        
        for class = 1:numOfClasses
            accuracy(class) = (TP(class) + TN(class)) / totalSamples;
            sensitivity(class) = TP(class) / (TP(class) + FN(class));
            specificity(class) = TN(class) / (FP(class) + TN(class));
            precision(class) = TP(class) / (TP(class) + FP(class));
            f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
        end
        
        field2 = 'accuracy';  value2 = accuracy;
        field3 = 'sensitivity';  value3 = sensitivity;
        field4 = 'specificity';  value4 = specificity;
        field5 = 'precision';  value5 = precision;
        field6 = 'recall';  value6 = sensitivity;
        field7 = 'Fscore';  value7 = f_score;
        stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);
        if exist('gorder','var')
            stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7,'groupOrder',gorder);
        end
        
    end
end
