%% Copyright @lucabotez

function [classes] = predict_classes(X, weights, ...
                  input_layer_size, hidden_layer_size, ...
                  output_layer_size)

  % spargem vectorul weights in cele 2 matrici corespunzatoare
  dim1 = hidden_layer_size * (input_layer_size + 1);
  Theta1 = reshape(weights'(1:dim1), hidden_layer_size, input_layer_size + 1);
  
  dim2 = dim1 + 1;
  Theta2 = reshape(weights'(dim2:end), output_layer_size, hidden_layer_size + 1);

  % salvam doar activarile ultimului layer al propagarii
  m = rows(X);
  [h, ~, ~] = forward_propagation(Theta1, Theta2, X, m);
  
  % aflam pozitia fiecarui maxim pt activari
  [~, classes] = max(h);
  classes = classes';

endfunction
