import * as tf from '@tensorflow/tfjs'

const MODEL_PATH = `https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json`
const MODEL_LS_PATH = `localstorage://my-tf-model`
let model

 async function loadModel() {
	console.log(await tf.io.listModels())
	try {
		model = await tf.loadLayersModel(MODEL_LS_PATH)
	} catch (e) {
		model = await tf.loadLayersModel(MODEL_PATH)
		await model.save(MODEL_LS_PATH)
	}
  	model.summary()

	const input = tf.tensor2d([[870]])
	const inputBatch = tf.tensor2d([[590], [750], [1200]]) 

	const result = model.predict(input)
	const resultBatch = model.predict(inputBatch)
	result.print()
	resultBatch.print()

	input.dispose()
	inputBatch.dispose()
	result.dispose()
	resultBatch.dispose()
	model.dispose()
 }
 
 loadModel()
