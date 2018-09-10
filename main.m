%% load MNIST data and label
train_inputs = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_inputs = loadMNISTImages('train-images.idx3-ubyte');
test_labels = loadMNISTLabels('train-labels.idx1-ubyte');

%% transform the labels to one-hot encoding format
train_targets = zeros(10, size(train_inputs,1));
for i = 1:1:size(train_inputs,2)
    train_targets(train_labels(i)+1,i) = 1;
end
test_targets = zeros(10, size(test_inputs,1));
for i = 1:1:size(test_inputs,2)
    test_targets(train_labels(i)+1,i) = 1;
end

%% configurations
img_size = size(train_inputs , 1);
num_units = [300 , 100 , 10];  % let's do 3-layer MLP first
lr = 0.01;
activation = 'sigmoid';
batch_size = 100;
epochs = 100;

%% weight initialiation
% images (layer-0) => W1 => layer-1 => W2 => layer-2 => W3 => output (layer-3)
% images (layer-0) <= B1 <= layer-1 <= B2 <= layer-2 <= B3 <= output (layer-3)
nLays = size(num_units , 2);
W = cell(1 , nLays);
B = cell(1 , nLays);
dW_sum = cell(1 , nLays);
for i = 1:1:nLays
    if i == 1
        W{1,i} = rand(num_units(i),img_size) / sqrt(num_units(i)) / sqrt(img_size);
        dW_sum{1,i} =  rand(num_units(i),img_size);
        B{1,i} = rand(img_size , num_units(i)) / sqrt(num_units(i)) / sqrt(img_size);
    else
        W{1,i} = rand(num_units(i),num_units(i-1)) / sqrt(num_units(i)) / sqrt(num_units(i-1));
        dW_sum{1,i} = rand(num_units(i),num_units(i-1));
        B{1,i} = rand(num_units(i-1),num_units(i)) / sqrt(num_units(i)) / sqrt(num_units(i-1));
    end
end

%% training 
iterations = floor(size(train_inputs,2) / batch_size);
train_ff_loss_record = zeros(1,epochs);
test_ff_loss_record = zeros(1,epochs);
for epoch = 1:1:epochs
    train_correct = 0;
    train_total = 0;
    test_correct = 0;
    total = 0;
    % training 
    for iter = 1:1:iterations
        for i = 1:1:nLays
            dW_sum{1,i} = zeros(size(dW_sum{1,i}));  
        end
        for batch_idx = 1:1:batch_size
            train_input = train_inputs(:,(iter-1)*batch_size+batch_idx);
            train_target = train_targets(:,(iter-1)*batch_size+batch_idx);
            train_states = classification_feedforward(train_input , W, 'activation', 'sigmoid');
            train_prediction = find(train_states{1,nLays}==max(train_states{1,nLays}));
            train_correct = train_correct + (train_target(train_prediction) > 0);
            train_total = train_total + 1;
            % calculate the classification loss and gradient
            [train_ff_loss, dW] = classification_feedback(train_input, train_states, train_target, W, B, 'activation', 'sigmoid');
            for i = 1:1:nLays
                dW_sum{1,i} = dW_sum{1,i} + dW{1,i};
            end
            train_ff_loss_record(1,epoch) = train_ff_loss_record(1,epoch) + train_ff_loss;
        end
        % optimization 
        for i = 1:1:nLays
            W{1,i} = W{1,i} - dW_sum{1,i} / batch_size * lr;
        end
    end
    train_ff_loss_record(1,epoch) = train_ff_loss_record(1,epoch) / iterations / batch_size;
    disp(['Epoch-' num2str(epoch) ': training loss is ' num2str(train_ff_loss_record(1,epoch)) ...
        '; The prediction accuracy is ' num2strr(train_correct/train_total)]);
    % testing
    for iter = 1:1:size(test_inputs,2)
        test_input = test_inputs(:,iter);
        test_target = test_targets(:,iter);
        test_states = classification_feedforward(test_input , W, 'activation', 'sigmoid');
        [test_ff_loss , ~] = classification_feedback(test_input, test_states, test_target, W, B, 'activation', 'sigmoid');
        test_ff_loss_record(1,epoch) = test_ff_loss_record(1,epoch) + test_ff_loss;
        test_prediction = find(test_states{1,nLays}==max(test_states{1,nLays}));
        test_correct = test_correct + (test_target(test_prediction) > 0);
    end
    test_ff_loss_record(1,epoch) = test_ff_loss_record(1,epoch) / size(test_inputs,2);
    disp(['Epoch-' num2str(epoch) ': testing loss is ' num2str(test_ff_loss_record(1,epoch)) ...
        '; The prediction accuracy is ' num2strr(test_correct/test_total)]);
end

    