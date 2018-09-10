% define the derivtive of sigmoid function
function y = dsigmoid(x)
y = x .* (1-x);
end