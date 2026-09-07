#!/usr/bin/env node
import * as fs from "node:fs";            // for createReadStream
import path from 'path';

// Oculus visemes (IN ORDER!)
export const visemes = [
  'aa', 'E', 'I', 'O', 'U',
  'PP', 'SS', 'TH', 'DD', 'FF',
  'kk', 'nn', 'RR', 'CH', 'sil'
];

export const visemeIds = {};
visemes.forEach( (x,i) => {
  visemeIds[x] = i;
});

// IPA to Oculus visemes
const ipaToOculusViseme = {
  // Silence / pause markers
  '': 'sil', 'ˈ': 'sil', 'ˌ': 'sil', '‖': 'sil', '|': 'sil',
  // Plosives / bilabials
  'p': 'PP', 'b': 'PP', 'm': 'PP',
  // Labiodentals
  'f': 'FF', 'v': 'FF',
  // Dentals
  'θ': 'TH', 'ð': 'TH',
  // Alveolar stops
  't': 'DD', 'd': 'DD',
  // Velar stops
  'k': 'kk', 'g': 'kk', 'q': 'kk', 'ɢ': 'kk',
  // Affricates
  'tʃ': 'CH', 'dʒ': 'CH', 'ts': 'CH', 'dz': 'CH',
  // Fricatives / sibilants
  's': 'SS', 'z': 'SS', 'ʃ': 'SS', 'ʒ': 'SS', 'ɕ': 'SS',
  'ʑ': 'SS', 'ç': 'SS', 'ʝ': 'SS', 'x': 'SS', 'ɣ': 'SS',
  'h': 'SS',
  // Nasals
  'n': 'nn', 'ŋ': 'nn', 'ɲ': 'nn', 'ɳ': 'nn', 'm̩': 'nn',
  // Liquids / approximants
  'ɹ': 'RR', 'r': 'RR', 'ɾ': 'RR', 'ɽ': 'RR', 'l': 'RR',
  'ɫ': 'RR', 'j': 'RR', 'w': 'RR',
  // Vowels – grouped by mouth shape
  // “aa” (open / low)
  'a': 'aa', 'aː': 'aa', 'ɑ': 'aa', 'ɑː': 'aa', 'ɐ': 'aa',
  'aɪ': 'aa', 'aʊ': 'aa', 'ä': 'aa',
  // “E” (mid)
  'ɛ': 'E', 'ɛː': 'E', 'e': 'E', 'eː': 'E', 'eɪ': 'E',
  'œ': 'E', 'ɜ': 'E', 'ʌ': 'E',
  // “I” (close front)
  'i': 'I', 'iː': 'I', 'ɪ': 'I', 'ɨ': 'I', 'y': 'I',
  'yː': 'I', 'ʏ': 'I',
  // “O” (mid back)
  'o': 'O', 'oː': 'O', 'ɔ': 'O', 'ɔː': 'O', 'ɒ': 'O',
  'ø': 'O', 'øː': 'O',
  // “U” (close back)
  'u': 'U', 'uː': 'U', 'ʊ': 'U', 'ɯ': 'U', 'ɯː': 'U',
  'ɤ': 'U',
  // Central vowels
  'ə': 'E', 'ɚ': 'E', 'ɘ': 'E'
};

// Create IPA-to-ID conversion object
export const ipaToVisemeID = {};
Object.keys(ipaToOculusViseme).forEach( key => {
  const value = ipaToOculusViseme[key];
  const id = visemes.indexOf(value);
  ipaToVisemeID[key] = (id >= 0) ? id : visemes.length;
});

/**
 * Find files either from a directory or a glob pattern.
 * 
 * @param {string} input Directory path or glob pattern (e.g., "./samples" or "./samples/*.wav")
 * @param {string[]} [extensions=null] File extensions to filter by, if null, no filter
 * @returns {Promise<string[]>} List of matching file paths (relative)
 */
export async function findFiles(input, extensions = null) {
  try {
    // Check if input is an existing directory
    const stat = await fs.promises.stat(input).catch(() => null);
    let files = [];

    if (stat && stat.isDirectory()) {
      // Read directory non-recursively
      const dirFiles = await fs.promises.readdir(input);
      files = dirFiles
        .filter(file => !extensions || extensions.includes(path.extname(file).toLowerCase()))
        .map(file => path.join(input, file));
    } else {
      // Treat input as a glob pattern
      const globFiles = fs.globSync(input, { nodir: true });
      files = globFiles.filter(file =>
        !extensions || extensions.includes(path.extname(file).toLowerCase())
      );
    }

    return files;
  } catch (err) {
    console.error('Error finding files:', err.message);
    return [];
  }
}

export async function fileExists(path) {
  try {
    await fs.promises.access(path);
    return true;
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    return false;
  }
}


/**
 * Asynchronously decode a 16-bit PCM mono WAV file into a Float32Array.
 * Returns both the Float32Array and the sampleRate.
 */
export async function getFloat32ArrayFromWav(filePath) {
  const buffer = await fs.promises.readFile(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  const getString = (v, start, len) =>
    Array.from({ length: len }, (_, i) =>
      String.fromCharCode(v.getUint8(start + i))
    ).join("");

  // Verify RIFF/WAVE header
  if (getString(view, 0, 4) !== "RIFF" || getString(view, 8, 4) !== "WAVE") {
    throw new Error("Invalid WAV file");
  }

  let offset = 12;
  let dataOffset, dataSize, sampleRate, numChannels, bitsPerSample;

  // Parse chunks
  while (offset < view.byteLength) {
    const chunkId = getString(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === "fmt ") {
      const audioFormat = view.getUint16(offset, true);
      numChannels = view.getUint16(offset + 2, true);
      sampleRate = view.getUint32(offset + 4, true);
      bitsPerSample = view.getUint16(offset + 14, true);

      if (audioFormat !== 1) throw new Error("Only PCM format supported");
      if (![16, 24, 32].includes(bitsPerSample)) {
        throw new Error("Only 16-, 24-, or 32-bit PCM supported");
      }
    } else if (chunkId === "data") {
      dataOffset = offset;
      dataSize = chunkSize;
      break;
    }

    offset += chunkSize;
  }

  if (!dataOffset) throw new Error("Missing 'data' chunk in WAV");
  if (!sampleRate) throw new Error("Missing 'fmt ' chunk in WAV");

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = dataSize / bytesPerSample;
  const samplesPerChannel = totalSamples / numChannels;

  const samples = new Float32Array(samplesPerChannel);
  let pos = dataOffset;

  // Decode and mix down to mono
  for (let i = 0; i < samplesPerChannel; i++) {
    let mixed = 0;

    for (let ch = 0; ch < numChannels; ch++) {
      let sample;

      if (bitsPerSample === 16) {
        sample = view.getInt16(pos, true) / 0x8000;
      } else if (bitsPerSample === 24) {
        // 24-bit little endian
        const b0 = view.getUint8(pos);
        const b1 = view.getUint8(pos + 1);
        const b2 = view.getUint8(pos + 2);
        let val = (b2 << 16) | (b1 << 8) | b0;
        if (val & 0x800000) val |= 0xff000000; // sign extend
        sample = val / 0x800000;
      } else if (bitsPerSample === 32) {
        sample = view.getInt32(pos, true) / 0x80000000;
      }

      mixed += sample;
      pos += bytesPerSample;
    }

    samples[i] = mixed / numChannels; // average channels → mono
  }

  return { sampleRate, samples };
}

export function floatTo16BitPCM(float32Array) {
  const buffer = Buffer.alloc(float32Array.length * 2); // 16-bit = 2 bytes
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i])); // clamp to [-1, 1]
    buffer.writeInt16LE(s < 0 ? s * 0x8000 : s * 0x7FFF, i * 2);
  }
  return buffer;
}


/**
 * Asynchronously read and parse a JSON file and
 * return an array of { v, t, d }.
 * 
 * @param {string} filePath Path to the JSON file.
 * @return {Promise<Object[]>} Visemes.
 */
export async function getDataFromJson(filePath) {
  try {
    const data = await fs.promises.readFile(filePath, "utf8");
    const o = JSON.parse(data);
    const parts = [];
    o.forEach( x => {
      const ps = x.ps || x.vs;
      ps.forEach( p => {
        let viseme = p.v;
        if ( viseme < 0 || viseme > 14 ) viseme = 14;
        let phoneme = p.p || ""+viseme;
        parts.push({ phoneme, viseme, t: p.t, d: p.d });
      });
    });
    return parts;
  } catch (err) {
    if (err.code === "ENOENT") {
      throw new Error(`File not found: ${filePath}`);
    } else if (err.name === "SyntaxError") {
      throw new Error(`Invalid JSON in file: ${filePath}`);
    } else {
      throw err;
    }
  }
}

/**
* Stream and yield sentences from a text file.
* Sentences are split on `.`, `!`, `?` followed by a space or EOF.
* Short sentences (< minLength) are merged with the next.
*/
export async function* readSentencesStream(filePath, minLength = 20) {
  const stream = fs.createReadStream(filePath, { encoding: "utf8" });

  let buffer = "";        // text buffer between chunks
  let carrySentence = ""; // holds a short sentence to merge with next

  for await (const chunk of stream) {
    buffer += chunk;

    // Split text into complete sentences (keep punctuation)
    const regex = /[^.!?]+[.!?]+(?:\s|$)/g;
    const matches = buffer.match(regex) || [];

    // If the chunk ended mid-sentence, keep the leftover
    const lastSentenceEnd = matches.reduce(
      (pos, s) => pos + s.length,
      0
    );
    buffer = buffer.slice(lastSentenceEnd); // keep leftover text

    for (let sentence of matches.map(s => s.trim())) {
      // Merge with carry-over if previous was too short
      if (carrySentence) {
        sentence = carrySentence + " " + sentence;
        carrySentence = "";
      }

      if (sentence.length < minLength) {
        carrySentence = sentence; // store to merge with next
      } else {
        yield sentence;
      }
    }
  }

  // Handle any leftover after stream ends
  if (carrySentence) yield carrySentence.trim();
  if (buffer.trim()) yield buffer.trim();
}

export function makeWavHeader(dataLength, sampleRate, channels, bitsPerSample) {
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * channels * bitsPerSample / 8;
  const blockAlign = channels * bitsPerSample / 8;

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataLength, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16); // fmt chunk size
  header.writeUInt16LE(1, 20);  // audio format = PCM
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(dataLength, 40);
  return header;
}

/**
* Builds a help message for a console app.
*
* @param {string} usage The usage line
* @param {Array<{ flag: string, description: string }>} options Array of option objects
* @param {string} [description] Optional general description of the app
* @return {string} Formatted help message
*/
export function buildHelpMessage(usage, options, description = '') {
  const lines = [];

  if (description) {
    lines.push(description.trim(), '');
  }

  lines.push('Usage:');
  lines.push(`  ${usage}`);
  lines.push('');

  if (options.length > 0) {
    lines.push('Options:');
    const maxFlagLength = Math.max(...options.map(opt => opt.flag.length));
    for (const opt of options) {
      const paddedFlag = opt.flag.padEnd(maxFlagLength + 2, ' ');
      lines.push(`  ${paddedFlag}${opt.description}`);
    }
  }

  lines.push('');

  return lines.join('\n');
}
