import * as tf from '@tensorflow/tfjs'

let model

// train()
load()

async function train() {
	const { TRAINING_DATA } = await import('https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js')
	// Grab a reference to the MNIST input values (pixel data).
	const INPUTS = TRAINING_DATA.inputs;
	// Grab reference to the MNIST output values.
	const OUTPUTS = TRAINING_DATA.outputs;
	// Shuffle the two arrays in the same way so inputs still match outputs indexes.
	tf.util.shuffleCombo(INPUTS, OUTPUTS);
	// Input feature Array is 1 dimensional.
	const INPUTS_TENSOR = tf.tensor2d(INPUTS);
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

	await save()
	await load()
}

async function save() {
	model.save('downloads://classification')
}

async function load() {
	model = await tf.loadLayersModel('/model/classification.json')
}

const PREDICTION_ELEMENT = document.getElementById('prediction')

async function evaluate() {
	const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select randon
	let answer = tf.tidy(function () {
		let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();
		let output = model.predict(newInput);
		output.print();
		return output.squeeze().argMax();
	})
	let index = await answer.array()
	// console.log(index)
	PREDICTION_ELEMENT.innerText = index;
	// PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong')
	answer.dispose();
	drawImage(INPUTS[OFFSET])
	// setTimeout(evaluate, 0)
}

async function evaluateFromCanvas() {
	const INPUT = getDataFromImage()
	drawImage(INPUT)
	let answer = tf.tidy(function () {
		let newInput = tf.tensor1d(INPUT).expandDims();
		let output = model.predict(newInput);
		output.print();
		return output.squeeze().argMax();
	})
	let index = await answer.array()
	PREDICTION_ELEMENT.innerText = index;
	PREDICTION_ELEMENT.setAttribute('class', 'correct')
	answer.dispose();
}


const CANVAS = document.getElementById('canvas')
const ctx = CANVAS.getContext('2d', { willReadFrequently: true })

function drawImage(digit) {
	const imageData = ctx.getImageData(0, 0, 28, 28)
	for (let i = 0; i < digit.length; i++) {
		imageData.data[i * 4 + 0] = digit[i] * 255
		imageData.data[i * 4 + 1] = digit[i] * 255
		imageData.data[i * 4 + 2] = digit[i] * 255
		imageData.data[i * 4 + 3] = 255
	}

	ctx.putImageData(imageData, 0, 0)
}

function clearImage() {
	ctx.clearRect(0, 0, 28, 28)
}

function getDataFromImage() {
	const output = []
	const imgData = ctx.getImageData(0, 0, 28, 28)
	for (let i=0; i<imgData.data.length / 4; i++) {
		const r = imgData.data[i * 4 + 0]
		const g = imgData.data[i * 4 + 1]
		const b = imgData.data[i * 4 + 2]
		const value = (r + g + b) / 3
		output.push(value / 255) // normalize value
	}
	return output
}

getDataFromImage()

const ZOOM = parseFloat(CANVAS.computedStyleMap().get('zoom').toString()) || 1
let dragging = false
ctx.lineWidth = 1
ctx.strokeStyle = 'white'
addEventListener('mousedown', e => {
	if (e.target.matches('canvas')) {
		const x = Math.round(e.offsetX / ZOOM)
		const y = Math.round(e.offsetY / ZOOM)
		dragging = true
		ctx.beginPath()
		ctx.moveTo(x, y)
		ctx.stroke()
	}
})
addEventListener('mousemove', e => {
	if (dragging) {
		const x = Math.round(e.offsetX / ZOOM)
		const y = Math.round(e.offsetY / ZOOM)
		ctx.lineTo(x, y)
		ctx.stroke()
	}
})
addEventListener('mouseup', e => {
	dragging = false
	evaluateFromCanvas()
})
addEventListener('mouseout', e => {
	dragging = false
	evaluateFromCanvas()
})
addEventListener('click', e => {
	if (e.target.matches('#clear')) {
		clearImage()
	}
})
addEventListener('keyup', e => {
	console.log(e.key)
	if (e.key === 'Backspace') {
		clearImage()
	}
})


// setInterval(evaluateFromCanvas, 100)

