package com.cocktail.mic;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Process;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * Native microphone capture for Cocktail (called from Unity C# via AndroidJavaObject).
 *
 * Why: Unity's Microphone API is processed mono. This opens an AudioRecord on the UNPROCESSED audio source
 * (falls back to VOICE_RECOGNITION, then MIC) and tries multi-channel capture (channel index mask), then reports
 * what the device actually gave us: channel count, sample rate, source, active microphones (position/orientation
 * when Android exposes them) and how timestamps are derived.
 *
 * Output: fixed-size chunks (chunkMs) of interleaved s16le PCM. poll() returns byte[] = 8-byte little-endian
 * capture time of the FIRST frame in microseconds (System.nanoTime()/1000, i.e. CLOCK_MONOTONIC) followed by the
 * PCM, or null when nothing is ready. C# converts the time to its own monotonic clock with a calibrated offset.
 *
 * Not tested on hardware by the author. API refs: android.media.AudioRecord / AudioRecord.Builder (API 23),
 * AudioRecord.getTimestamp(AudioTimestamp,int) (API 24), AudioRecord.getActiveMicrophones (API 29),
 * MediaRecorder.AudioSource.UNPROCESSED (API 24).
 */
public class CocktailMic {
    private static final String TAG = "CocktailMic";

    private AudioRecord rec;
    private Thread thread;
    private volatile boolean running;
    private final ArrayBlockingQueue<byte[]> queue = new ArrayBlockingQueue<byte[]>(250); // 5 s of 20 ms chunks

    private int sampleRate;
    private int channels;
    private int chunkFrames;
    private int sourceId;
    private String sourceName = "";
    private String tsMode = "none";
    private volatile long overruns;
    private volatile long chunksProduced;
    private volatile long readErrors;
    private volatile String lastError = "";
    private String micsJson = "[]";

    /** System.nanoTime() (CLOCK_MONOTONIC on Android). Used by C# to calibrate its clock. */
    public long nowNs() {
        return System.nanoTime();
    }

    /**
     * Start capture. Tries wantChannels..1 channels, sources UNPROCESSED -> VOICE_RECOGNITION -> MIC, and rates
     * preferredRate -> 48000 -> 44100 -> 16000. Returns a JSON status string ("ok":true on success).
     */
    /** Test switch: use only this MediaRecorder.AudioSource id (e.g. 5 = CAMCORDER); -1 = default order. */
    private volatile int forcedSource = -1;

    public void setForcedSource(int src) {
        forcedSource = src;
    }

    public synchronized String start(int wantChannels, int preferredRate, int chunkMs, boolean preferUnprocessed) {
        stop();
        int[] sources = forcedSource >= 0
                ? new int[]{forcedSource}
                : preferUnprocessed
                ? new int[]{MediaRecorder.AudioSource.UNPROCESSED, MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC}
                : new int[]{MediaRecorder.AudioSource.MIC};
        int[] rates = new int[]{preferredRate, 48000, 44100, 16000};
        StringBuilder tried = new StringBuilder();

        for (int src : sources) {
            for (int ch = Math.max(1, wantChannels); ch >= 1; ch--) {
                for (int rate : rates) {
                    AudioRecord r = tryOpen(src, ch, rate, chunkMs, tried);
                    if (r != null) {
                        rec = r;
                        sampleRate = rate;
                        channels = r.getChannelCount();
                        sourceId = src;
                        sourceName = sourceName(src);
                        chunkFrames = Math.max(1, rate * chunkMs / 1000);
                        try {
                            r.startRecording();
                        } catch (Throwable t) {
                            lastError = "startRecording: " + t;
                            r.release();
                            rec = null;
                            tried.append(sourceName(src)).append('/').append(ch).append("ch/").append(rate).append(": startRecording failed; ");
                            continue;
                        }
                        if (r.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
                            tried.append(sourceName(src)).append('/').append(ch).append("ch/").append(rate).append(": not recording; ");
                            r.release();
                            rec = null;
                            continue;
                        }
                        micsJson = describeMics(r);
                        running = true;
                        thread = new Thread(new Runnable() {
                            @Override
                            public void run() {
                                captureLoop();
                            }
                        }, "CocktailMicCapture");
                        thread.start();
                        return status(true, tried.toString());
                    }
                }
            }
        }
        lastError = "no AudioRecord configuration could be opened";
        return status(false, tried.toString());
    }

    private AudioRecord tryOpen(int src, int ch, int rate, int chunkMs, StringBuilder tried) {
        try {
            AudioFormat.Builder fb = new AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(rate);
            if (ch == 1) {
                fb.setChannelMask(AudioFormat.CHANNEL_IN_MONO);
            } else if (ch == 2) {
                fb.setChannelMask(AudioFormat.CHANNEL_IN_STEREO);
            } else {
                fb.setChannelIndexMask((1 << ch) - 1);
            }
            int minBuf = AudioRecord.getMinBufferSize(rate,
                    ch == 1 ? AudioFormat.CHANNEL_IN_MONO : AudioFormat.CHANNEL_IN_STEREO,
                    AudioFormat.ENCODING_PCM_16BIT);
            if (minBuf <= 0) minBuf = rate / 10 * 2;
            if (ch > 2) minBuf = minBuf * ch / 2;
            int chunkBytes = Math.max(1, rate * chunkMs / 1000) * ch * 2;
            int bufBytes = Math.max(minBuf * 2, chunkBytes * 8);
            AudioRecord r = new AudioRecord.Builder()
                    .setAudioSource(src)
                    .setAudioFormat(fb.build())
                    .setBufferSizeInBytes(bufBytes)
                    .build();
            if (r.getState() != AudioRecord.STATE_INITIALIZED) {
                tried.append(sourceName(src)).append('/').append(ch).append("ch/").append(rate).append(": not initialized; ");
                r.release();
                return null;
            }
            if (r.getChannelCount() != ch) {
                tried.append(sourceName(src)).append('/').append(ch).append("ch/").append(rate)
                        .append(": got ").append(r.getChannelCount()).append("ch; ");
                // Accept what the device gave us only if it is at least mono (caller reads r.getChannelCount()).
            }
            return r;
        } catch (Throwable t) {
            tried.append(sourceName(src)).append('/').append(ch).append("ch/").append(rate).append(": ")
                    .append(t.getClass().getSimpleName()).append(' ').append(t.getMessage()).append("; ");
            return null;
        }
    }

    private void captureLoop() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        } catch (Throwable ignored) {
        }
        final int bytesPerFrame = channels * 2;
        final int chunkBytes = chunkFrames * bytesPerFrame;
        long framesRead = 0;
        // Fallback timing: capture time of frame 0, estimated as min(arrival - framesRead/rate).
        long t0EstNs = Long.MAX_VALUE;
        AudioTimestamp ts = new AudioTimestamp();
        boolean haveTs = false;

        while (running) {
            byte[] out = new byte[8 + chunkBytes];
            int got = 0;
            while (got < chunkBytes && running) {
                int n = rec.read(out, 8 + got, chunkBytes - got, AudioRecord.READ_BLOCKING);
                if (n < 0) {
                    readErrors++;
                    lastError = "read returned " + n;
                    try {
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        return;
                    }
                    continue;
                }
                got += n;
            }
            if (!running) break;
            long arrivalNs = System.nanoTime();
            long chunkStartFrame = framesRead;
            framesRead += chunkFrames;

            long firstFrameNs;
            boolean ok = false;
            if (Build.VERSION.SDK_INT >= 24) {
                try {
                    ok = rec.getTimestamp(ts, AudioTimestamp.TIMEBASE_MONOTONIC) == AudioRecord.SUCCESS;
                } catch (Throwable t) {
                    ok = false;
                }
            }
            long cand = arrivalNs - framesRead * 1000000000L / sampleRate;
            if (cand < t0EstNs) t0EstNs = cand;
            if (ok) {
                // ts says: frame #framePosition was captured at nanoTime.
                firstFrameNs = ts.nanoTime - (ts.framePosition - chunkStartFrame) * 1000000000L / sampleRate;
                if (!haveTs) {
                    haveTs = true;
                    tsMode = "AudioRecord.getTimestamp(MONOTONIC)";
                }
            } else {
                firstFrameNs = t0EstNs + chunkStartFrame * 1000000000L / sampleRate;
                if (!haveTs) tsMode = "arrival-min-estimate";
            }
            long us = firstFrameNs / 1000L;
            ByteBuffer.wrap(out, 0, 8).order(ByteOrder.LITTLE_ENDIAN).putLong(us);
            if (!queue.offer(out)) {
                queue.poll();
                queue.offer(out);
                overruns++;
            }
            chunksProduced++;
        }
    }

    /** Next chunk (8-byte LE microsecond timestamp + s16le PCM) or null. Non-blocking. */
    public byte[] poll() {
        return queue.poll();
    }

    public int queued() {
        return queue.size();
    }

    public synchronized void stop() {
        running = false;
        if (thread != null) {
            try {
                thread.join(500);
            } catch (InterruptedException ignored) {
            }
            thread = null;
        }
        if (rec != null) {
            try {
                rec.stop();
            } catch (Throwable ignored) {
            }
            rec.release();
            rec = null;
        }
        queue.clear();
    }

    public String statusJson() {
        return status(rec != null, "");
    }

    private String status(boolean ok, String tried) {
        return "{\"ok\":" + ok
                + ",\"rate\":" + sampleRate
                + ",\"channels\":" + channels
                + ",\"chunk_frames\":" + chunkFrames
                + ",\"source\":\"" + sourceName + "\""
                + ",\"source_id\":" + sourceId
                + ",\"raw\":" + (ok && sourceId == MediaRecorder.AudioSource.UNPROCESSED)
                + ",\"ts_mode\":\"" + esc(tsMode) + "\""
                + ",\"overruns\":" + overruns
                + ",\"chunks\":" + chunksProduced
                + ",\"read_errors\":" + readErrors
                + ",\"error\":\"" + esc(lastError) + "\""
                + ",\"tried\":\"" + esc(tried) + "\""
                + ",\"api\":" + Build.VERSION.SDK_INT
                + ",\"mics\":" + micsJson + "}";
    }

    private static String sourceName(int src) {
        if (src == MediaRecorder.AudioSource.UNPROCESSED) return "UNPROCESSED";
        if (src == MediaRecorder.AudioSource.VOICE_RECOGNITION) return "VOICE_RECOGNITION";
        if (src == MediaRecorder.AudioSource.MIC) return "MIC";
        return "SOURCE_" + src;
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "'").replace("\n", " ").replace("\r", " ");
    }

    /** Active microphone descriptions (API 29+): lets the hub/locate code know array geometry if Android exposes it. */
    private static String describeMics(AudioRecord r) {
        if (Build.VERSION.SDK_INT < 29) return "[]";
        try {
            java.util.List<android.media.MicrophoneInfo> list = r.getActiveMicrophones();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                android.media.MicrophoneInfo m = list.get(i);
                if (i > 0) sb.append(',');
                android.media.MicrophoneInfo.Coordinate3F p = m.getPosition();
                android.media.MicrophoneInfo.Coordinate3F o = m.getOrientation();
                sb.append("{\"id\":\"").append(esc(m.getDescription())).append('"')
                        .append(",\"location\":").append(m.getLocation())
                        .append(",\"group\":").append(m.getGroup())
                        .append(",\"index_in_group\":").append(m.getIndexInTheGroup())
                        .append(",\"directionality\":").append(m.getDirectionality())
                        .append(",\"pos_m\":[").append(p.x).append(',').append(p.y).append(',').append(p.z).append(']')
                        .append(",\"orient\":[").append(o.x).append(',').append(o.y).append(',').append(o.z).append(']')
                        .append('}');
            }
            return sb.append(']').toString();
        } catch (Throwable t) {
            return "[]";
        }
    }
}
