%% Copyright @lucabotez

function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)
  
  % spargem vectorul params in cele 2 matrici corespunzatoare
  dim1 = hidden_layer_size * (input_layer_size + 1);
  Theta1 = reshape(params'(1:dim1), hidden_layer_size, input_layer_size + 1);
  
  dim2 = dim1 + 1;
  Theta2 = reshape(params'(dim2:end), output_layer_size, hidden_layer_size + 1);
  
  % extindem vectorul y la o matrice ce contine pe fiecare linie
  % elemente ale bazei canonice pentru R^output_layer_size
  Y = eye(output_layer_size);
  Y = Y(:, y);
  
  % salvam rezultatul propagarii
  m = rows(X);
  [h, a_2, a_1] = forward_propagation(Theta1, Theta2, X, m);
  
  % calculam valoarea functiei de cost, unde sum1, sum2 si sum3
  % sunt variabile auxiliare ce calculeaza parti din formula
  sum1 = sum(sum(-Y .* log(h) - (1 .- Y ) .* log(1 .- h)));
  sum2 = sum(sum(Theta1(:, 2:end) .^ 2));
  sum3 = sum(sum(Theta2(:, 2:end) .^ 2));
  
  J = sum1 / m + (lambda * (sum2 + sum3)) / (2 * m);
  
  % aplicam back propagation pentru a determina gradientii
  delta_3 = h - Y;
  Delta2 = delta_3 * a_2';
  
  aux1 = Theta2' * delta_3;
  aux2 = a_2 .* (1 .- a_2);
  delta_2 = aux1 .* aux2;
  delta_2 = delta_2(2:end, :);
  
  Delta1 = delta_2 * a_1';
  
  grad_1 = Delta1 ./ m + ...
          ([zeros(size(Theta1, 1), 1) Theta1(:, 2:end)] .* lambda) ./ m;
  grad_2 = Delta2 ./ m + ...
          ([zeros(size(Theta2, 1), 1) Theta2(:, 2:end)] .* lambda) ./ m;
  
  grad = [grad_1(:); grad_2(:)];

endfunction