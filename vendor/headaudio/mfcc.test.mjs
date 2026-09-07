
import { describe, test, expect } from '@jest/globals';
import * as PARAMS from "../modules/parameters.mjs";
import { MFCC } from '../modules/mfcc.mjs';

describe('Performance', () => {
  
  let mfcc = new MFCC();

  // Test data
  const M = 1000;
  const blocks = Array.from({ length: M }, () => {
    return Array.from({ length: PARAMS.MFCC_SAMPLES_N }, () => {
      return Math.random() * 2 - 1; // Range [-1, 1]
    });
  });
  
  test('compute', () => {
    const start = process.hrtime.bigint();
    blocks.forEach( x => {
      mfcc.compute(x);
    });
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / 1_000_000 / M;
    console.log(`MFCC computation took ${durationMs.toFixed(3)} ms per block.`);
    expect(durationMs).toBeLessThan(0.2);
  });

});
