import * as tf from '@tensorflow/tfjs-node-gpu';
import { getAlphabetArray, getRandomImagesFromBin, normalizeData} from './helpers/helpers.js';
import fs from 'fs'

let correctionSrt = "";

let sId = 1;
const start = async (isInit = false) => {
    sId = fs.readdirSync('models').map(item => item.split("-")[1]).reduce((p,c) => {
        const n = Number.parseInt(c);
        return n > p ? n : p;
    }, 0) + 1

    const letters = getAlphabetArray()
    const lettersData = normalizeData(letters.map(l => {
        if (correctionSrt.includes(l.letter)) {
            return getRandomImagesFromBin(l, 300)
        }
        return getRandomImagesFromBin(l, 35)
    }));

    const images = [];
    const labels = [];
    const numClasses = 26;

    for (const data of lettersData) {
        for (const image of data.images) {
            images.push(image);
            labels.push(data.l.letter.charCodeAt(0)-97);
        }
    }

    const imagesNested = images.map((image) => {
        const rows = [];
        for (let i = 0; i < 32; i++) {
            const row = [];
            for (let j = 0; j < 32; j++) {
                row.push([image[i * 32 + j]]);
            }
            rows.push(row);
        }
        return rows;
    })

    const alphabetImagesNested = lettersData.map((data, index) => {
        const image = data.images[index];
        const rows = [];
        for (let i = 0; i < 32; i++) {
            const row = [];
            for (let j = 0; j < 32; j++) {
                row.push([image[i * 32 + j]]);
            }
            rows.push(row);
        }
        return rows;
    });

    const trainImages = tf.tensor4d(imagesNested, [imagesNested.length, 32, 32, 1]);

    const labelsTensor = tf.tensor1d(labels, 'int32');
    const trainLabels = tf.oneHot(labelsTensor, numClasses);

    let model = null;
    if (isInit) {
        model = tf.sequential();
        console.log('Model created')
        model.add(tf.layers.conv2d({
            inputShape: [32, 32, 1],
            filters: 32,
            kernelSize: 3,
            activation: "relu",
        }));
        model.add(tf.layers.dropout({dtype: "float32", rate: 0.1}))
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: "relu",
        }));
        model.add(tf.layers.dropout({dtype: "float32", rate: 0.1}))
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: 3,
            activation: "relu",
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
            units: 256,
            activation: "relu"
        }));
        model.add(tf.layers.dropout({dtype: "float32", rate: 0.1}))
        model.add(tf.layers.dense({
            units: 128,
            activation: "relu"
        }));
        model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
    } else {
        model = await tf.loadLayersModel(`file://./models/model-${sId-1}/model.json`)
    }

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    const history = await model.fit(trainImages, trainLabels, {
        epochs: 5,
        batchSize: 45,
        validationSplit: 0.1,
        callbacks: {
            onEpochEnd: async () => {
                const alph = 'abcdefghijklmnopqrstuvwxyz';
                const alphabetImagesTensor = tf.tensor4d(alphabetImagesNested, [alphabetImagesNested.length, 32, 32, 1]);
                const alphabetPredictions = model.predict(alphabetImagesTensor);
                const alphabetPredictionLabels = alphabetPredictions.argMax(-1);

                const alphabetPredictionLabelsArray = alphabetPredictionLabels.arraySync();
                correctionSrt = "";
                console.log(alphabetPredictionLabelsArray.map((val, index) => {
                    if(alph[val] !== alph[index]) {
                        correctionSrt = `${correctionSrt}${alph[index]}`
                    }
                    return [alph[val], alph[index]]
                }));
                alphabetImagesTensor.dispose();
                alphabetPredictions.dispose();
                alphabetPredictionLabels.dispose();
                sId++;
            }
        }
    });

    await model.save(`file://./models/model-${sId}`);
    console.log(correctionSrt)
    setTimeout(() => {
        start(false).then(res => {})
    }, 1000)
}

start().then(res => {})