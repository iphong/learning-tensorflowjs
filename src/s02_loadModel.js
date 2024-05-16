import * as tf from '@tensorflow/tfjs'

let movenet
const IMG = document.getElementById('exampleImg')
async function loadModel() {
	movenet = await tf.loadGraphModel("/model/model.json")
	
	let exampleInputTensor = tf.zeros([1, 192, 192, 3], 'int32')
	const imageTensor = tf.browser.fromPixels(IMG)
	console.log(imageTensor.shape)

	let cropStartPoint = [15, 170, 0]
	let cropSize = [345, 345, 3]
	let croppedTensor = tf. slice(imageTensor, cropStartPoint, cropSize)
	let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true) .toInt()
	console. log (resizedTensor.shape)

	let tensorOutput = movenet.predict(tf.expandDims(resizedTensor))
	let arrayOutput = await tensorOutput.array()
	console. log (arrayOutput) ;

	exampleInputTensor.dispose()
	imageTensor.dispose()
	resizedTensor.dispose()
	croppedTensor.dispose()
	tensorOutput.dispose()
} 

// const z = tf.zeros([1, 192, 192, 3])
// console.log(z)

loadModel()
