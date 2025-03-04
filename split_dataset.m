%% Copyright @lucabotez
function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % generam un set de permutari aleatoare
  index = randperm(rows(X));
  
  % rotunjim procentul
  number = floor(percent * rows(X));

  % amestecam matricea de teste si vectorul de clase
  X = X(index, :);
  y = y(index, :);

  % impartim rezultatul
  X_train = X(1:number, :);
  X_test = X(number+1:rows(X), :);

  y_train = y(1:number, 1);
  y_test = y(number + 1: rows(y), 1);
endfunction
