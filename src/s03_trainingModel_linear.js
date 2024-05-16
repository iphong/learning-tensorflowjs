import * as tf from '@tensorflow/tfjs'
import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';

window.tf = tf
// Input feature pairs (House size, Number of Bedrooms)
const INPUTS = TRAINING_DATA.inputs;
// Current listed house prices in dollars given their features above (target output values you to predict).
const OUTPUTS = TRAINING_DATA.outputs;
// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS)
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)

const FEATURE_RESULTS = normalize(INPUTS_TENSOR)
FEATURE_RESULTS.MIN_VALUES.print()
FEATURE_RESULTS.MAX_VALUES.print()
FEATURE_RESULTS.NORMALIZED_VALUES.print()

INPUTS_TENSOR.dispose()

// Now actually create and define model architecture.
const model = tf.sequential();
// We will use one dense layer with 1 neuron (units) and an input of
// 2 input feature values (representing house size and number of rooms)
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));
model.summary();

train()

async function train() {
	const LEARNING_RATE = 0.01

	model.compile({
		optimizer: tf.train.sgd(LEARNING_RATE),
		loss: 'meanSquaredError'
	})

	// Finally do the training itself.
	let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
		validationSplit: 0.15, // Take aside 15% of the data to use for validation testing.
		shuffle: true, // Ensure data is shuffled in case it was in an order
		batchSize: 64, // As we have a lot of training data, batch size is set to 64.
		epochs: 10 // Go over the data 10 times!
	})

	OUTPUTS_TENSOR.dispose()
	FEATURE_RESULTS.NORMALIZED_VALUES.dispose()

	console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
	console.log ("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));

	evaluate()
}

function evaluate() {
	// Predict answer for a single piece of data.
	tf.tidy(function() {
		let newInput = normalize(tf.tensor2d([[750, 111]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES)
		let output = model.predict(newInput.NORMALIZED_VALUES) ;
		output.print();
	});

	// Finally when you no longer need to make any more predictions,
	// clean up remaining Tensors.
	FEATURE_RESULTS. MIN_VALUES.dispose();
	FEATURE_RESULTS.MAX_VALUES.dispose();
	model.dispose();
	console.log(tf.memory().numTensors)
}
function normalize(tensor, min, max) {
	const result = tf.tidy(function () {
		// Find the minimum value contained in the Tensor.
		const MIN_VALUES = min || tf.min(tensor, 0);
		// Find the maximum value contained in the Tensor.
		const MAX_VALUES = max || tf.max(tensor, 0);
		// Now subtract the MIN_VALUE from every value in the Tensor
		// And store the results in a new Tensor.
		const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
		// Calculate the range size of possible values.
		const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
		// Calculate the adjusted values divided by the range size as a new Tensor.
		const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

		return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
	})
	return result;
}
