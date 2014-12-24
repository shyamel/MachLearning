"Logistic Regression"


function newton_weights  = newton_train(training_X, training_Y, number_of_inputs, iteration_number)
	"with Newton's method"
	h_weights = rand(1, number_of_inputs);
	grad = zeros(1, number_of_inputs);
	hessian = zeros(1,1);


	#Calculate weights
	for iterations = 1:iteration_number

		for sample = 1: rows(training_X)

			received_pizza = training_Y(sample);
			sig = 1/(1+e^(-h_weights*training_X(sample,:)'));
			diff = sig - received_pizza;
			grad = grad + diff * training_X(sample,:)/ rows(training_X);
			hessian = hessian + (sig * (1  - sig) * training_X(sample,:)' * training_X(sample,:))/ rows(training_X);
		
		endfor

		h_weights = h_weights - ctranspose(pinv(hessian) * ctranspose(grad));

	endfor
	newton_weights = h_weights;


endfunction


function unreg_weights = unreg_train(training_X, training_Y, number_of_inputs, iteration_number)
	"with Stochastic Gradient Descent"
	unreg_weights = rand(1, number_of_inputs);
	learning_rate = 0.1;
	iteration_number = 10;

	#Calculate weights
	for iterations = 1:iteration_number
		for sample = 1: rows(training_X)
			for feature = 1:number_of_inputs
				received_pizza = training_Y(sample);
				diff = received_pizza - 1/(1+e^(-unreg_weights*training_X(sample,:)'));
				unreg_weights(feature) = unreg_weights(feature) + learning_rate* diff * training_X(sample,feature);
			endfor
		endfor
	endfor

endfunction

function reg_weights = reg_train(training_X, training_Y, number_of_inputs, iteration_number)
	"with Regularized Stochastic Gradient Descent"
	reg_weights = rand(1, number_of_inputs);
	learning_rate = 0.1;
	iteration_number = 10;

	#Calculate weights
	for iterations = 1:iteration_number
		for sample = 1: rows(training_X)
			for feature = 1:number_of_inputs
				received_pizza = training_Y(sample);
				diff = received_pizza - 1/(1+e^(-reg_weights* training_X(sample,:)'));
				reg_weights(feature) = reg_weights(feature) + learning_rate* (-0.01*reg_weights(feature) + diff * training_X(sample,feature));
			endfor
		endfor
	endfor

endfunction


function correct_percent = test(weights, test_X, test_Y)
	num_error = 0;
	for row = 1: rows(test_X)
		predict = 0;
		p = 1/(1+e^(-weights * test_X(row,:)'));
		if(p > 0.5)
			predict = 1;
		endif
		if( predict != test_Y(row))
			num_error = num_error +1;
		endif
	
	endfor

	correct_percent = 100 - num_error/rows(test_X) * 100

endfunction

function x_matrix = pad_with_ones(data, last_index_of_testing_samples, number_of_inputs, input_column_start, input_column_end)
	X_init = data(1:last_index_of_testing_samples, input_column_start:input_column_end + 1);

	x_matrix = ones(last_index_of_testing_samples, number_of_inputs + 1);

	for row = 1:rows(X_init)
		for col = 2: number_of_inputs + 1
			x_matrix(row, col) = X_init(row, col - 1);
		endfor
	endfor
endfunction

function x_mat = feature_scaling(X, last_index_of_testing_samples, number_of_inputs)

	for col = 1:number_of_inputs
		max_X = max(X(:,col));
		min_X = min(X(:, col));
		if(max_X != min_X)
			for row = 1:last_index_of_testing_samples
				X(row,col) = (X(row, col) - min_X)/(max_X - min_X);
			endfor
		endif
	endfor
	x_mat = X;
endfunction

function main(filename)
	data= load(filename);

	last_index_of_training_samples = 4040;
	last_index_of_testing_samples = rows(data);
	output_column = columns(data);
	input_column_start = 2;
	input_column_end = columns(data) - 1;
	number_of_inputs = input_column_end - input_column_start +1;
	iteration_number = 15;

	x_matrix = pad_with_ones(data, last_index_of_testing_samples, number_of_inputs, input_column_start, input_column_end);
	number_of_inputs = number_of_inputs +1;

	X = feature_scaling(x_matrix, last_index_of_testing_samples, number_of_inputs);


	training_X = X(1:last_index_of_training_samples, 1:number_of_inputs);
	training_Y = data(1:last_index_of_training_samples, output_column);

	training = [training_X training_Y];
	training = training(randperm(length(training)),:);

	training_X = training(1:last_index_of_training_samples, 1:number_of_inputs);
	training_Y = training(1:last_index_of_training_samples, output_column);

	test_X = X(last_index_of_training_samples + 1: last_index_of_testing_samples, 1:number_of_inputs);
	test_Y = data(last_index_of_training_samples + 1: last_index_of_testing_samples, output_column);

	#weights = unreg_train(training_X, training_Y, number_of_inputs, iteration_number);
	#test(weights, test_X, test_Y);
	#weights = reg_newton_train(training_X, training_Y, number_of_inputs, iteration_number);
	#test(weights, test_X, test_Y);
	weights = newton_train(training_X, training_Y, number_of_inputs, iteration_number);

	test(weights, test_X, test_Y);



endfunction

main('pizza.csv');




