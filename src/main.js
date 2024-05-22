import * as tf from '@tensorflow/tfjs'
import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js'
const { inputs: INPUTS, outputs: OUTPUTS } = TRAINING_DATA
const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs)

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

const model = tf.sequential()

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 16,
  kernelSize: 3, // Square filter of 3 by 3, could also be a rectangle 2 by 3
  strides: 1, // goes through every single pixel in the image
  padding: 'same', // to calculate pixels at the edge of the image, that doesn't have pixels around it.
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3, // Square filter of 3 by 3, could also be a rectangle 2 by 3
  strides: 1, // goes through every single pixel in the image
  padding: 'same', // to calculate pixels at the edge of the image, that doesn't have pixels around it.
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

model.summary()

train()

const PREDICTION_ELEMENT = document.getElementById('prediction')

async function train() {
  model.compile({
    optimizer: 'adam', // Adam changes the learning rate over time which is useful.
    loss: 'categoricalCrossentropy', // As this is a classification problem, dont use MSE.
    metrics: ['accuracy']
  })
  // reshape input from array of number to a batch of 28,28,1
  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);
  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true, // Ensure data is shuffled again before using each time.
    validationSplit: 0.15,
    epochs: 30, // Go over the data 30 times!
    batchSize: 256,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch, '->', logs)
      }
    }
  });
  RESHAPED_INPUTS.dispose();
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate();
}

function evaluate() {
  const OFFSET = Math.floor((Math.random() * INPUTS.length));
  let answer = tf.tidy(function() {
    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);
    let output = model.predict(newInput.reshape([1, 28, 28, 1]));
    output.print();
    return output.squeeze().argMax();
  })
  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = LOOKUP[index];
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    // drawImage(INPUTS[OFFSET]);
  })
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
