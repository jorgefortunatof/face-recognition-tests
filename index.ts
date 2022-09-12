const canvas = require("canvas");
const faceapi = require("face-api.js");

faceapi.env.monkeyPatch({ Canvas: canvas.Canvas, Image: canvas.Image, ImageData: canvas.ImageData })

async function loadLabeledImages() {
	const descriptions: any[] = [];

	const SHELDON = [
		'./images/sheldon/sheldon1.png',
		'./images/sheldon/sheldon2.png',
		'./images/sheldon/sheldon3.png',
		'./images/sheldon/sheldon4.png',
		'./images/sheldon/sheldon5.png'
	];

	for(let imagePath of SHELDON){
		const image = await canvas.loadImage(imagePath);
		const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor()

		descriptions.push(detection.descriptor);
	}

	return new faceapi.LabeledFaceDescriptors('Sheldon', descriptions)
}

async function run () {
	await Promise.all([
			faceapi.nets.ssdMobilenetv1.loadFromDisk('./models'),
			faceapi.nets.faceRecognitionNet.loadFromDisk('./models'),
			faceapi.nets.faceLandmark68Net.loadFromDisk('./models')
	])
	
	const SHELDON = './images/sheldon/compare.png'
	const sheldon = await canvas.loadImage(SHELDON)

	const labeledDescriptors = await loadLabeledImages()
	const faceMatcher = new faceapi.FaceMatcher([labeledDescriptors], 0.6)

	const detection = await faceapi.detectSingleFace(sheldon).withFaceLandmarks().withFaceDescriptor()
	const result = faceMatcher.findBestMatch(detection.descriptor)

	console.log(result)
}

run()

