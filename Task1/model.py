'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...] 	 	
	outputs = model.calculate_output(sample)

	Inputs Names: 	
	1 )sepal_length
	2 )sepal_width
	3 )petal_length
	4 ) petal_width

You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]], np.int32)	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
'''

import numpy as np

class NeuralNetwork:
 
	def __init__(self):
 
		self.parameters_number = 75
 
	def scaling_layer(self,inputs):

		outputs = [None] * 4

		outputs[0] = (inputs[0]-5.843329906)/0.8280659914
		outputs[1] = (inputs[1]-3.053999901)/0.4335939884
		outputs[2] = (inputs[2]-3.758670092)/1.764420033
		outputs[3] = (inputs[3]-1.19867003)/0.7631610036

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 9

		combinations[0] = 0.311149 +0.212714*inputs[0] -0.246406*inputs[1] +0.451339*inputs[2] +0.383649*inputs[3] 
		combinations[1] = 1.04875 +0.187369*inputs[0] -0.0618301*inputs[1] -0.887339*inputs[2] -1.04888*inputs[3] 
		combinations[2] = 0.311996 +0.20603*inputs[0] -0.244952*inputs[1] +0.453892*inputs[2] +0.393238*inputs[3] 
		combinations[3] = -0.531124 -0.0526154*inputs[0] -1.04609*inputs[1] +0.854773*inputs[2] +0.351947*inputs[3] 
		combinations[4] = -0.32101 -0.22425*inputs[0] +0.242037*inputs[1] -0.469606*inputs[2] -0.382011*inputs[3] 
		combinations[5] = 0.294097 +0.193946*inputs[0] -0.25412*inputs[1] +0.448971*inputs[2] +0.400635*inputs[3] 
		combinations[6] = 0.324435 +0.208327*inputs[0] -0.245706*inputs[1] +0.472093*inputs[2] +0.392537*inputs[3] 
		combinations[7] = 0.965306 +0.170135*inputs[0] -0.071614*inputs[1] -0.830352*inputs[2] -0.986609*inputs[3] 
		combinations[8] = 1.00378 +0.178492*inputs[0] -0.0677249*inputs[1] -0.857134*inputs[2] -1.01536*inputs[3] 
		
		activations = [None] * 9

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])
		activations[6] = np.tanh(combinations[6])
		activations[7] = np.tanh(combinations[7])
		activations[8] = np.tanh(combinations[8])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 3

		combinations[0] = 0.0939058 -0.739234*inputs[0] +0.411133*inputs[1] -0.744149*inputs[2] -0.624206*inputs[3] +0.739931*inputs[4] -0.748885*inputs[5] -0.752674*inputs[6] +0.427613*inputs[7] +0.419404*inputs[8] 
		combinations[1] = 0.0168138 +0.498375*inputs[0] +0.998832*inputs[1] +0.496825*inputs[2] -0.519348*inputs[3] -0.507961*inputs[4] +0.483064*inputs[5] +0.519898*inputs[6] +0.89796*inputs[7] +0.949883*inputs[8] 
		combinations[2] = -0.110662 +0.240458*inputs[0] -1.41055*inputs[1] +0.246733*inputs[2] +1.14343*inputs[3] -0.231997*inputs[4] +0.264322*inputs[5] +0.231996*inputs[6] -1.32438*inputs[7] -1.36846*inputs[8] 
		
		activations = [None] * 3

		sum_ = 0;

		sum_ = 	np.exp(combinations[0]) + 	np.exp(combinations[1]) + 	np.exp(combinations[2]);

		activations[0] = np.exp(combinations[0])/sum_;
		activations[1] = np.exp(combinations[1])/sum_;
		activations[2] = np.exp(combinations[2])/sum_;

		return activations;


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

		return output_probabilistic_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

			output = np.append(output,output_probabilistic_layer, axis=0)

		return output
