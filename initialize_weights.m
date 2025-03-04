%% Copyright @lucabotez

function [matrix] = initialize_weights(L_prev, L_next)
  epsilon = sqrt(6) / sqrt(L_prev + L_next);
  
  % matricea va avea elemente din intervalul (-epsilon, epsilon)
  matrix = rand(L_next, L_prev + 1) * epsilon;
endfunction
