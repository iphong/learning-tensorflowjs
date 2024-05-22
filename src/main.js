import * as tf from '@tensorflow/tfjs'
import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js'
const { inputs: INPUTS } = TRAINING_DATA
// Map output index to label
const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ];

console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs)

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
// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

INPUTS_TENSOR.print()
