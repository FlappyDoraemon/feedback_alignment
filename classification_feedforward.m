function states = classification_feedforward(input , W, varargin)

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
    activation_function = @sigmoid;
end
    
%% calculation
nLays = size(W , 2);
states = cell(1,nLays);
% feedforward propogation;
for i = 1:1:nLays
    if i == 1
        states{1,i} = activation_function(W{1,i} * input);
    else
        states{1,i} = activation_function(W{1,i} * states{1,i-1});
    end
end

end

