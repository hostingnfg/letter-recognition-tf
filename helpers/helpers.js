import fs from "fs";
import * as path from "path";
import Jimp from "jimp";


export function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
}

export function letterToHex (letter) {
    const code = letter.charCodeAt(0);
    return  code.toString(16);
}

export function getAlphabetArray() {
    const alphabet = [];
    for (let i = 97; i <= 122; i++) {
        const l = String.fromCharCode(i);
        alphabet.push({
            letter: l,
            hex: letterToHex(l)
        });
    }
    return alphabet;
}

export function getImagesList(letterHex)
{
    const dirPath = `BY_CLASS/${letterHex}/`;
    const files = fs.readdirSync(dirPath)
    const dirs = files.filter(f => {
        const stat =  fs.statSync(path.join(dirPath, f));
        if (!stat.isFile()) {
            return true;
        }
        return false;
    });
    return dirs.reduce((p,c) => {
        fs.readdirSync(`${dirPath}/${c}`).map(f => `${dirPath}${c}/${f}`).forEach(e => p.push(e))
        return p;
    }, [])
}

export function getResizedImagesList(letter, size)
{
    const dirPath = `images/${size}/${letter}`;
    const files = fs.readdirSync(dirPath)
    return files.reduce((p,c) => {
        p.push(`${dirPath}/${c}`);
        return p;
    }, [])
}

export function getJSONImageData(letter, size)
{
    return JSON.parse(fs.readFileSync(`../imagesData/${letter}-${size}.json`).toString())
}

export function getRandomImagesFromBin(l, count)
{
    const data = fs.readFileSync(`dataL/${l.letter}.bin`)

    let randomIndices = new Set();
    while (randomIndices.size < count && randomIndices.size < (data.length - 1) / 1024) {
        const randomIndex = Math.floor(Math.random() * data.length / 1024);
        randomIndices.add(randomIndex);
    }

    const images = Array.from(randomIndices).map(index => {
        const pixels = data.slice(index, index+1024)
        return pixels.map(item => {
            if (Math.random() > 0.995) {
                return item
            }
            if (Math.random() > 0.99) {
                return getRandomInt(1,254)
            }
            return 255 - item
        })
    })

    return {l, images}
}

export function getRandomElements(arr, count) {
    if (count >= arr.length) {
        return arr;
    }

    let randomIndices = new Set();

    while (randomIndices.size < count) {
        const randomIndex = Math.floor(Math.random() * arr.length);
        randomIndices.add(randomIndex);
    }

    return Array.from(randomIndices).map(index => arr[index]);
}

export function normalizeData(data)
{
    return data.map(({l, images}) => {
        return {
            l,
            images: images.map(img => {
                return img.map(pixel => pixel/255);
            })
        }
    })
}

export async function getImageData(path, h, w)
{
    const image = await Jimp.read(path);
    const pixels = [];
    for(let ih = 0; ih<h; ih++)
    {
        for (let iw = 0; iw<w; iw++) {
            const pixel = Jimp.intToRGBA(image.getPixelColor(ih, iw));
            pixels.push(((pixel.r + pixel.g + pixel.b) / 3))
        }
    }
    return pixels;
}

export async function loadImageAndPreprocess(imagePath, targetWidth, targetHeight, {letter}, index) {
    const image = await Jimp.read(imagePath);
    image.resize(32, 32).grayscale();
    const pixels = new Uint8Array(targetWidth * targetHeight);
    for (let y = 0; y < targetHeight; y++) {
        for (let x = 0; x < targetWidth; x++) {
            const idx = y * targetWidth + x;
            const pixel = Jimp.intToRGBA(image.getPixelColor(x, y));
            pixels[idx] = (pixel.r + pixel.g + pixel.b) / 3;
        }
    }
    return Float32Array.from(Float32Array.from(pixels).map((pixel) => ((pixel / 255) + 0.1)/1.2));
}