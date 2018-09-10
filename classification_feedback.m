function [ff_loss, dW] = classification_feedback(input, states, target, W, B, varargin)

%% Note on the derivative of different activation functions
% for the following functions, the x is the output result after activation function
% x = activation(x_in)
% y(x) = d(x)/d(x_in)

%% Keywords
% default
activation = 'sigmoid';
% load
for iter = 1:2:size(varargin,2) 
    Keyword = varargin{iter};
    Value   = varargin{iter+1};
    if strcmpi(Keyword,'activation')
        activation = Value; % 
    else
        warning(['classification_feedforward(): unknown keyword ' Keyword]);
    end
end

%% select the activation function
if strcmpi(activation , 'sigmoid')
    dactivation_function = @dsigmoid;
end

%% store the pseudo-derivative of states or weights
dstates = cell(1,size(states,2));
dW = cell(1,size(dstates,2));    

%% calculation
nLays = size(W, 2);
ff_loss = sum((states{1,nLays}-target).*(states{1,nLays}-target));

for i = size(dstates,2):-1:1
    if i == size(dstates,2)
        dstates{1,i} = dactivation_function(states{1,i}).*(states{1,i}-target);
        dW{1,i} = dstates{1,i} * states{1,i-1}';
    elseif i > 1
        dstates{1,i} = dactivation_function(states{1,i}).*(B{1,i+1}*dstates{1,i+1});
        dW{1,i} = dstates{1,i} * states{1,i-1}';
    else % i==1
        dstates{1,i} = dactivation_function(states{1,i}).*(B{1,i+1}*dstates{1,i+1});
        dW{1,i} = dstates{1,i} * input';
    end
end

end
