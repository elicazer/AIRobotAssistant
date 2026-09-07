#!/usr/bin/env node
import { Processor } from "../modules/processor.mjs";
import * as utils from './training-node-utils.mjs';
import * as fs from 'node:fs';

// Constants
const SAMPLE_RATE = 16000;
const CHANNELS = 1;
const BITS_PER_SAMPLE = 16;

// Default options
let input = null;
let output = null;
let override = false;
let help = false;

// Help message
const helpText = utils.buildHelpMessage(
  'node downsample.mjs [options] -i <file> -o <file>',
  [
    { flag: '-h, --help', description: 'Show this help message.' },
    { flag: '-i, --input <file>', description: 'Input file (.wav).' },
    { flag: '-o, --output <file>', description: 'Output file (.wav).' },
    { flag: '--override', description: 'Replace the output file if it exists.' },
  ],
  "Downsamples audio (.wav) into 16kHz audio."
);

// Parse command-line arguments
const args = process.argv.slice(2);
args.forEach((arg, index) => {
  const next = args[index + 1];
  if ( (arg === '--input' || arg === '-i') && next ) {
    input = next;
  } else if ( (arg === '--output' || arg === '-o')  && next ) {
    output = next;
  } else if (arg === '--override' ) {
    override = true;
  } else if (arg === '--help' || arg === '-h' ) {
    help = true;
  }
});
if ( !output || !input || help ) {
  console.log(helpText);
  process.exit(1);
}

// Check if output file already exists
const outputFileExists = await utils.fileExists(output);
if ( outputFileExists && !override ) {
  console.log(`Output file already exists. Use --override option to override.`);
  process.exit(1);
}

const wavFiles = await utils.findFiles(input, [".wav"]);
const N = wavFiles.length;
if (N === 0) {
  console.log('Audio file not found.');
  process.exit(1);
}

const wavFile = wavFiles[0];
const { sampleRate, samples } = await utils.getFloat32ArrayFromWav(wavFile);

// Init HeadLipSyncProcessor
const processor = new Processor({
  processorOptions: {
    sampleRate: sampleRate, // Source sample rate
    frameEventsEnabled: true
  },
});

const pcmChunks = [];
processor.port.postMessage = (o) => {
  if ( o.event === "frame" ) {
    const pcm = utils.floatTo16BitPCM(o.frame);
    pcmChunks.push(pcm);
  }
}


try {
  
  // Process
  const S = samples.length;
  for( let j=0; j<S; j+=128 ) {
    const block = [[new Float32Array(samples.buffer, j * 4, Math.min(128, S-j) )]];
    processor.process(block,[],parameters);
  }

  // Save file
  if ( pcmChunks.length ) {
    const pcmData = Buffer.concat(pcmChunks);
    const header = utils.makeWavHeader(pcmData.length, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE);
    const wavBuffer = Buffer.concat([header, pcmData]);
    await fs.promises.writeFile(output, wavBuffer);
    console.log(`✅ OK! Saved ${output}.`);
  }

} catch(err) {
  console.error('❌ Error in processing:', err.message);
}

