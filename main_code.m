data = load('mnist_data.mat');

X1 = cast(data.trainImages,'double');
X1 = X1/255;
L = data.trainLabels;
x1 = zeros(20,784);

node_no = 512;

x2 = zeros(1,node_no);
h2 = zeros(1,node_no);
x3 = zeros(1,10);
h3 = zeros(1,10);

l2 =zeros(1,10); 

eta = 0.01;


w12 = (0 + (1).*rand(node_no,784))/sqrt(784);
w12 = w12 - mean(w12,2)*ones(1,784);

w23 = (0 + (1).*rand(10,node_no))/sqrt(node_no);
w23 = w23 - mean(w23,2)*ones(1,node_no);

b12 = zeros(node_no,1);
b23 = zeros(10,1);


del2 = zeros(1,node_no);
del3 = zeros(1,10);
del3_s = zeros(1,10);


epoch_tot = 4000;

train_Acc= zeros(1,epoch_tot);
test_Acc = zeros(1,epoch_tot);

err_test = zeros(1,epoch_tot);
err_train = zeros(1,epoch_tot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:epoch_tot
    
    
    count = 0;
    for iter = 1:40

        x1(:,:) = X1((iter-1)*20+1:iter*20,:);
        l = L((iter-1)*20+1:iter*20);
     
        del3 = zeros(1,10);
        del3_s = zeros(1,10);

        dw23 = zeros(10,node_no);
        db23 = zeros(10,1);
        
        for i = 1:20
         
            l2 = zeros(1,10);
         
            h2 = x1(i,:)*w12' - b12';
            %x2 = 1./(1+exp(-1*h2));
            x2 = Relu(h2);
    
            h3 = x2(1,:)*w23' - b23';
            x3 = exp(h3);
            x3 = x3/sum(x3);
        
            if (x3(l(i)+1) == max(x3))
                count = count+1; 
            end
        
            l2(1,l(1,i)+1) = l2(1,l(1,i)+1) + 1;
            err_train(epoch) = err_train(epoch) - log(x3(l(i)+1));
            
            del3 = (x3-l2);
            del3_s = del3_s + del3;
            dw23 = dw23 + (0.01)*del3'*x2(1,:);
            db23 = db23 + (0.01)*del3';
        end
  
        w23 = w23 - dw23/20;
        b23 = b23 + db23/20;
        del3 = del3_s/20;
 
        %del2 = del3*w23 .* (exp(-1*h2)./((1+exp(-1*h2)).^2));
        del2 = del3*w23 .* ReluD(h2);
        dw12 = (0.01)*del2'*(x1(i,:));
        w12 = w12 + dw12;
        b12 = b12 - (0.01)*del2';
    end

    
    
    X2 = cast(data.testImages,'double');
    X2 = X2/255;
    L2 = data.testLabels;
    count2 = 0;
    for i = 1:200
    
        h2 = X2(i,:)*w12' - b12';
        %x2 = 1./(1+exp(-1*h2));
        x2 = Relu(h2);
    
        h3 = x2(1,:)*w23' - b23';
        x3 = exp(h3);
        x3 = x3/sum(x3);
    
        if (x3(L2(i)+1) == max(x3))
            count2 = count2+1; 
        end
        
        err_test(epoch) = err_test(epoch) - log(x3(L2(i)+1));
    
    end
    
    train_Acc(epoch) = (count/800)*100;
    test_Acc(epoch) = (count2/200)*100;
    disp([epoch,train_Acc(epoch),test_Acc(epoch)]);
    
    err_test(epoch) = err_test(epoch)/200;
    err_train(epoch) = err_train(epoch)/800;
    
end

figure;
plot(1:epoch_tot,train_Acc);
xlabel('epoch no.');
ylabel('Training data Accuracy (%)');

figure;
plot(1:epoch_tot,test_Acc);
xlabel('epoch no.');
ylabel('Testing data Accuracy (%)');

figure;
plot(1:epoch_tot,err_train);
xlabel('epoch no.');
ylabel('Training data error');

figure;
plot(1:epoch_tot,err_test);
xlabel('epoch no.');
ylabel('Testing data error');


disp([epoch,train_Acc(epoch),test_Acc(epoch),err_train(epoch),err_test(epoch)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    


