import * as tf from '@tensorflow/tfjs'

const INPUTS = []
for (let i=0; i<=20; i++) {
	INPUTS.push(i)
}

const OUTPUTS = []
for (let i=0; i<INPUTS.length; i++) {
	OUTPUTS.push(INPUTS[i] * INPUTS[i])
	// OUTPUTS.push(INPUTS[i] * Math.abs(INPUTS[i]))
}

const INPUTS_TENSOR = tf.tensor1d(INPUTS)
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)


const FEATURE_RESULTS = normalize(INPUTS_TENSOR)
FEATURE_RESULTS.MIN_VALUES.print()
FEATURE_RESULTS.MAX_VALUES.print()
FEATURE_RESULTS.NORMALIZED_VALUES.print()

INPUTS_TENSOR.dispose()

// Now actually create and define model architecture.
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));
model.summary();

const LEARNING_RATE = 0.0001
const OPTIMIZER = tf.train.sgd(LEARNING_RATE)

train()

async function train() {

	model.compile({
		optimizer: OPTIMIZER,
		loss: 'meanSquaredError'
	})

	// Finally do the training itself.
	let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
		// validationSplit: 0.15, // Take aside 15% of the data to use for validation testing.
		shuffle: true, // Ensure data is shuffled in case it was in an order
		batchSize: 2, // As we have a lot of training data, batch size is set to 64.
		epochs: 200, // Go over the data 200 times!
		callbacks: {
			onEpochEnd: logProgress
		}
	})

	OUTPUTS_TENSOR.dispose()
	FEATURE_RESULTS.NORMALIZED_VALUES.dispose()

	console.log(results.history)
	console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
	// console.log ("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));

	evaluate()
}

function logProgress(epoch, logs) {
	console.log('EPOCH:', epoch, Math.sqrt(logs.loss))
	// if (epoch == 140) {
	// 	OPTIMIZER.setLearningRate(LEARNING_RATE / 2)
	// }
}

function evaluate() {
	// Predict answer for a single piece of data.
	tf.tidy(function() {
		let newInput = normalize(tf.tensor1d([5]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES)
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
