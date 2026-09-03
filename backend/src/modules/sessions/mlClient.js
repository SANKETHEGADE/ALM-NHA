/**
 * ML Service HTTP Client
 * Forwards audio to Sanket's /analyze endpoint
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const ML_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || '15000', 10);

/**
 * Returns mock ML response adhering to Sanket's frozen contract JSON schema
 */
function getMockMlResponse(sessionId) {
  return {
    session_id: sessionId || 'sess_mock123',
    duration_sec: 8.4,
    transcript: {
      text: "Someone help, there's smoke coming from the kitchen!",
      language: "en",
      confidence: 0.91
    },
    sound_events: [
      { label: "smoke_alarm", confidence: 0.87, start_sec: 1.2, end_sec: 3.0 },
      { label: "shouting", confidence: 0.78, start_sec: 0.0, end_sec: 2.1 }
    ],
    emotion: {
      primary: "fear",
      confidence: 0.82,
      arousal: "high"
    },
    speakers: {
      count: 1,
      diarization: [
        { speaker_id: "spk_1", start_sec: 0.0, end_sec: 8.4 }
      ]
    },
    model_insight: {
      label: "fire_emergency",
      confidence: 0.89
    },
    reasoning: {
      trace: "High-arousal fear speech co-occurs with a smoke alarm sound.",
      summary: "A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency.",
      severity_hint: "critical"
    },
    processing_ms: 1840
  };
}

/**
 * Call POST /analyze on ML Service
 * Implements retry-once-then-fail on 500 or transient errors
 *
 * @param {Object} options
 * @param {string} options.sessionId
 * @param {Buffer|Blob|Stream} options.audioBuffer
 * @param {string} [options.filename='audio.wav']
 * @param {string} [options.mimeType='audio/wav']
 * @param {string} [options.languageHint]
 * @returns {Promise<Object>} ML response JSON
 */
async function analyzeAudio({ sessionId, audioBuffer, filename = 'audio.wav', mimeType = 'audio/wav', languageHint }) {
  if (!sessionId) {
    throw new Error('sessionId is required for ML analysis');
  }

  // Support local development / offline mocking
  if (process.env.MOCK_ML === 'true') {
    // Simulate brief processing delay
    await new Promise((resolve) => setTimeout(resolve, 300));
    return getMockMlResponse(sessionId);
  }

  let attempt = 0;
  const maxAttempts = 2; // initial + 1 retry on 500

  while (attempt < maxAttempts) {
    attempt++;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ML_TIMEOUT_MS);

    try {
      // Build multipart/form-data using standard Node FormData/Blob or FormData polyfill
      const formData = new FormData();
      const audioBlob = new Blob([audioBuffer || Buffer.from('mock audio')], { type: mimeType });
      formData.append('audio_file', audioBlob, filename);
      formData.append('session_id', sessionId);
      if (languageHint) {
        formData.append('language_hint', languageHint);
      }

      const targetUrl = `${ML_SERVICE_URL.replace(/\/+$/, '')}/analyze`;
      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        return await response.json();
      }

      // If status is 500 and we have attempts remaining, retry once
      if (response.status === 500 && attempt < maxAttempts) {
        await new Promise((res) => setTimeout(res, 500));
        continue;
      }

      const errorText = await response.text().catch(() => '');
      const err = new Error(`ML Service error (${response.status}): ${errorText}`);
      err.status = response.status;
      throw err;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        const timeoutErr = new Error(`ML Service timed out after ${ML_TIMEOUT_MS}ms`);
        timeoutErr.code = 'ETIMEDOUT';
        throw timeoutErr;
      }

      // Retry once on network connection refusal or 500
      if (attempt < maxAttempts && (error.code === 'ECONNREFUSED' || error.status === 500)) {
        await new Promise((res) => setTimeout(res, 500));
        continue;
      }

      throw error;
    }
  }
}

module.exports = {
  analyzeAudio,
  getMockMlResponse,
  ML_SERVICE_URL,
  ML_TIMEOUT_MS
};
