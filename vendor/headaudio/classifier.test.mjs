
import { describe, test, expect } from '@jest/globals';
import * as PARAMS from '../modules/parameters.mjs';
import { Classifier } from '../modules/classifier.mjs';
import { Training } from '../modules/training.mjs';
import * as fs from 'node:fs';

// Classifier instance
const classifier = new Classifier();
const training = new Training();

// Load binary model
const bin = await fs.promises.readFile('./dist/model-en-mixed.bin');
const buffer = bin.buffer.slice( bin.byteOffset, bin.byteOffset + bin.byteLength );

// Extract records
const model = [];
const len = buffer.byteLength;
let pos = 0;
while( (pos + PARAMS.RECORD_OFFSET) <= len ) {
  const record = new Float32Array(buffer, pos, PARAMS.RECORD_LEN);
  const p = training.decodeBinaryRecord(record);
  model.push(p);
  pos += PARAMS.RECORD_OFFSET;
}

describe('Predict', () => {
  
  // Import model to classifier
  classifier.import({ reset: true, model });

  // CSV labels
  let csvContent = [ '""', ...model.map( x => '"' + x.phoneme + '"' ) ].join(",");

  test('predict - mean vectors', () => {

    const N = model.length;
    for( let i=0; i<N; i++ ) {
      const prototype = model[i];
      const viseme = prototype.viseme;
      let prediction;
      for( let j=0; j<10; j++ ) {
        prediction = classifier.predict(prototype.mu);
      }
      if ( viseme === PARAMS.MODEL_VISEME_SIL ) {
        expect(prediction.viseme).toBe(null);
      } else {
        expect(prediction.viseme).toBe(viseme);
      }
      expect(prediction.distances[i]).toBeLessThan(1e-5);

      // Add to CSV content
      csvContent += '\n"' + prototype.phoneme + '",';
      csvContent += prediction.distances.map( x => x.toFixed(1) ).join(",");
    }

    // Write to CSV file
    fs.writeFileSync("./tests/distances.csv", csvContent);
    console.log("CSV file written!");

  });

});

describe('Performance', () => {
  
  // Test vectors
  const M = 1000;
  const vs = Array.from({ length: M }, () => {
    return Array.from({ length: PARAMS.MFCC_COEFF_N_WITH_DELTAS }, () => {
      return 2 * (Math.random() - 0.5);
    });
  });

  // Import model to classifier
  classifier.import({ reset: true, model });

  test('predict', () => {
    const start = process.hrtime.bigint();
    for( let i=0; i<M; i++ ) {
      const p = classifier.predict(vs[i]);
    }
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / 1_000_000 / M;
    console.log(`Model prediction took ${durationMs.toFixed(3)} ms per vector.`);
    expect(durationMs).toBeLessThan(0.2);
  });

});
