// Runs on the audio rendering thread. Taps the raw mic samples and hands
// each ~128-sample quantum to the main thread untouched (Float32, still at
// whatever rate the AudioContext runs at) -- all format conversion
// (resample/Int16) happens in app.js, off the audio thread.
class PcmRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channelData = inputs[0] && inputs[0][0];
    if (channelData && channelData.length > 0) {
      // Copy out: the Float32Array backing this call is reused by the
      // engine on the next quantum, so postMessage must clone, not alias.
      this.port.postMessage(channelData.slice());
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorderProcessor);
