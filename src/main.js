import * as tf from '@tensorflow/tfjs'
import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js'
const { inputs: INPUTS, outputs: OUTPUTS } = TRAINING_DATA
const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ];

console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs)


async function train() {
	// Shuffle the two arrays in the same way so inputs still match outputs indexes.
	tf.util.shuffleCombo(INPUTS, OUTPUTS);
	
  // Input feature Array is 2 dimensional.
  const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
	// Output feature Array is 1 dimensional.
	const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

	model = tf.sequential()

	model.add(tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }))
	model.add(tf.layers.dense({ units: 32, activation: 'relu' }))
	model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

	model.summary()

	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	})

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
		shuffle: true,
		validationSplit: 0.2,
		batchSize: 512,
		epochs: 200,
		callbacks: {
			onEpochEnd: (epoch, logs) => {
				console.log('epoch', epoch, '-->', logs)
			}
		}
	})

	INPUTS_TENSOR.dispose()
	OUTPUTS_TENSOR.dispose()
}

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = tf.scalar(min)
    const MAX_VALUES = tf.scalar(max)
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES)
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES)
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE)
    return NORMALIZED_VALUES
  })
  return result
}

INPUTS_TENSOR.print()
