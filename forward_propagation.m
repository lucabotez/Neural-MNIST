%% Copyright @lucabotez

function [h, a_2, a_1] = forward_propagation(Theta1, Theta2, X, m)
  % functie auxiliara ce realizeaza forward propagation
  % calculand toate activarile deodata
  
  % primul layer de activari (testele din matricea X + biasul adaugat
  % la inceput)
  a_1 = ones(m, 1);
  a_1 = [a_1, X];
  a_1 = a_1';

  % activarile din layerul hidden + biasul considerat
  z_2 = Theta1 * a_1;
  a_2 = ones(1, columns(z_2));
  a_2 = [a_2; sigmoid(z_2)];

  % rezultatul propagarii
  z_3 = Theta2 * a_2;
  h = sigmoid(z_3);
endfunction