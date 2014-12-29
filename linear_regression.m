"Linear Regression with Basis function and OLS"



function basis = basis_function(X, num_rows, polynomial_degree)
	basis = ones(num_rows, polynomial_degree + 1);
	
	for i = 1:num_rows #fill in our basis function matrix
	
		for j = 2:polynomial_degree+1 #because no need to mess with first row
		
			basis(i, j) = (X(i,1))^(j-1);
			
		endfor
		
	endfor

endfunction




function weight = ols_fn(basis_matrix, Y)
	weight = inverse(basis_matrix' * basis_matrix) * basis_matrix' * Y # use closed form for OLS to generate appropriate weights
endfunction


function rmse = training_error(weights, basis_matrix, Y)
	mse = 0;
	for sample = 1: rows(basis_matrix)
		mse = mse + (weights * ctranspose(basis_matrix(sample))- Y(sample))^2;
	endfor
	mse = mse / rows(basis_matrix);
	rmse = mse^(1/2)
	
endfunction


function main(filename)

	data = load(filename);	
	
	attribute_number = 4;
	polynomial_degree = 6;
	num_rows = rows(data);
	
	X = data(:, attribute_number);
	Y = data(:, 1);

	basis_matrix = basis_function(X, num_rows, polynomial_degree);

	'Weights'	
	weights = ols_fn(basis_matrix, Y);

	training_error(weights, basis_matrix, Y);


endfunction

main('auto-mpg.data');
	
	
