# Autonomous audio-visual sound discovery for Cocktail

Status: proposed implementation specification and research expansion, 2026-10-25.
Repository baseline: `3b9878ff7f63ac148245bbf085eb17b83843042e` (`origin/main` when inspected).
Scope of this change: documentation. Proposed modules, commands, metrics and thresholds below are not implemented or measured by this change.

## 1. Product outcome

Turn a recorded scene into an inspectable set of persistent sound sources. Each source has a timeline, an isolated audio estimate, a stable identity, a visual association when supported, and an explanation of the evidence behind that association. People and objects light up when their own sound is active. Selecting an object plays its stem; selecting an off-screen source plays its sound without inventing a visible location.

Preserve the original ambition: discover semantic sound events, distinguish one talker from several overlapping talkers, segment and track visual instances with SAM 3, associate acoustic and visual identities, isolate with SAM Audio, and export both an annotated MP4 and an interactive multi-stem player. Build the recorded-video path first, then connect its tested components to Cocktail's live hub.

The central addition is a **persistent sound scene graph with bounded refinement**. Discovery proposes candidates, association proposes links, separation produces candidate evidence, and validation revises those links. A single greedy decision must not permanently attach every later bark, voice or chirp to the first plausible object.

Keep four identities distinct:

- **Event:** a time interval such as a bark or a speaking turn.
- **Acoustic source:** the persistent emitter across events, such as speaker A or an unidentified dog.
- **Visual track:** one visible instance through time, including its masklets and visibility gaps.
- **Association:** a time-bounded, evidence-backed link between an acoustic source and a visual track.

A person can talk, laugh and cough without becoming three people. A television can reproduce several voices while remaining one physical sound emitter. Two visible cats are two visual candidates; that fact alone does not establish two acoustic sources.

## 2. Fit the existing repository

Cocktail already uses one Python 3.12 project in `hub/`, an optional GPU dependency group, existing separation/localization/tracking modules, a Unity client in `quest/`, and evaluation tools in `eval/`. Extend these boundaries instead of creating an independent `av_sound_segmenter/` project with another lockfile.

Implementation grounding, inspected at the baseline above:

| Existing location | Existing responsibility | Planned reuse or extension |
|---|---|---|
| [`hub/pyproject.toml`](../hub/pyproject.toml) | `cocktail-hub`; Python `>=3.12,<3.13`; Hatch package list; CUDA 12.8 wheel index | Add explicit optional AV dependencies and new package registrations after compatibility checks. Preserve CPU-only installation. |
| [`hub/ingest/buffers.py`](../hub/ingest/buffers.py) | 16 kHz audio ring, pose history, newest video frame | Reuse timestamp conventions. Add a bounded AV window buffer separately from the existing newest-frame path. |
| [`hub/ingest/server.py`](../hub/ingest/server.py) | Audio resampling, epoch/gap handling, sensor ingestion | Provide immutable window snapshots; keep model inference out of callbacks. |
| [`hub/mockclient/session.py`](../hub/mockclient/session.py) | `session.json`, multichannel `audio.wav`, `pose.jsonl`, timestamp-named JPEGs, optional reference stems | Use as the recorded-session input adapter. Preserve its existing format; put new AV metadata in a sidecar. |
| [`hub/separate/models.py`](../hub/separate/models.py) | Two-output `VoiceSeparator`; AST `SoundLabeler.probs()` exposes 527 label scores | Reuse AST for broad discovery and the separator as a proposal generator. Do not infer source count from output slots. |
| [`hub/separate/live.py`](../hub/separate/live.py) | Overlap-add, stream ordering, speech gates, per-mic estimates, `StreamChunk` | Reuse timing and reconstruction lessons. Introduce a separate variable-source result contract. |
| [`hub/separate/voice_memory.py`](../hub/separate/voice_memory.py) | Voice profile continuity | Reuse cautiously after profile-quality and checkpoint-license checks. Avoid learning from uncertain mixtures. |
| [`hub/locate/pair.py`](../hub/locate/pair.py) and [`tdoa.py`](../hub/locate/tdoa.py) | Calibrated microphone-pair delay estimation | Extend through a spatial-evidence adapter; preserve calibration and sign conventions. |
| [`hub/track/tracker.py`](../hub/track/tracker.py) | Stable source IDs from observations, lifecycle and bearing matching | Bridge confirmed scene sources into this public lifecycle. Add tests before changing same-bearing duplicate behavior. |
| [`hub/events/server.py`](../hub/events/server.py) | Latest-only events, `Tracker` ownership, freshness and epoch handling | Single publication point; prevent separate live and offline workers from publishing conflicting identities. |
| [`hub/protocol/messages.py`](../hub/protocol/messages.py) | Frozen client protocol, `Source`, `TargetUpdated`, `AudioOut` | Project compatible fields; keep richer research artifacts internal. |
| [`docs/QUEST_PRD.txt`](QUEST_PRD.txt), section 14 | Folder owners: `vision/`, `who/`, `replay/`, etc. | Use those reserved module names even where implementation has not landed. |
| [`eval/`](../eval/) | Separation benchmarks | Add AV scene fixtures and scoring, with real recordings distinguished from synthetic fixtures. |

Some architecture/status prose predates current code. For example, `events/server.py` already owns a `Tracker`; do not schedule that wiring again solely because an older paragraph says it is missing. Likewise, the current live separator defaults its direction merge gate to off. Inspect implementation and current PRs before integration.

### 2.1 Proposed file layout

This tree is a destination map, not a request to generate empty files. Existing files are marked `[existing]`; other entries are proposed and should appear only with a working phase.

```text
Cocktail/
├── ARCHITECTURE.md                       [existing: link to this proposal]
├── docs/
│   ├── QUEST_PRD.txt                     [existing: base product requirements]
│   └── AV_SOUND_DISCOVERY_SPEC.md        this specification
├── hub/
│   ├── pyproject.toml                    [existing: single project]
│   ├── uv.lock                           [existing: shared dependency lock]
│   ├── av_scene/                         NEW orchestration and internal artifacts
│   │   ├── __main__.py                   run / preflight / inspect / serve commands
│   │   ├── pipeline.py                   stage DAG and bounded refinement
│   │   ├── schemas.py                    internal versioned data contracts
│   │   ├── cache.py                      hashes, atomic writes, resume/invalidation
│   │   ├── runtime.py                    GPU leases and model lifecycle
│   │   ├── preflight.py                  runtime/checkpoint capability report
│   │   ├── protocol_bridge.py            scene source -> existing wire models
│   │   └── configs/                      packaged TOML defaults and array example
│   ├── ingest/
│   │   ├── media_io.py                   MP4 + WAV ingest and timestamp mapping
│   │   └── av_window.py                  bounded raw-audio/video snapshots
│   ├── separate/
│   │   ├── models.py                     [existing AST + voice proposal baseline]
│   │   ├── discovery.py                  AST proposals, taxonomy and noun mapping
│   │   ├── pe_events.py                  PE-A-Frame adapter and time spans
│   │   └── diarization.py                regular diarization and anchor intervals
│   ├── vision/                           PRD C5 ownership
│   │   ├── sam3_tracker.py               image/video backends and session lifetime
│   │   ├── mask_store.py                 RLE, geometry and track stitching
│   │   └── face_tracks.py                face crops, visibility, landmark evidence
│   ├── who/                              PRD E2 ownership
│   │   ├── asd.py                        LR-ASD preprocessing and scores
│   │   ├── pe_correspondence.py          crop/audio correspondence evidence
│   │   ├── matcher.py                    gated matching with unmatched options
│   │   └── refinement.py                 candidate extraction and reassociation
│   ├── locate/
│   │   ├── pair.py, tdoa.py              [existing calibrated pair estimator]
│   │   └── av_spatial.py                 camera rays and optional SRP-PHAT backend
│   ├── replay/                           PRD D3 ownership
│   │   ├── sam_audio.py                  prompted extraction adapter
│   │   ├── chunked.py                    context windows and overlap-add
│   │   └── residual.py                   duplicate/leakage and remainder analysis
│   ├── render/                           NEW output ownership
│   │   ├── envelope.py                   activity gates and display envelopes
│   │   ├── video.py                      timestamped mask compositor
│   │   ├── viewer.py                     bundle creation + local serving
│   │   └── assets/                       HTML, CSS, WebAudio player
│   ├── protocol/, track/, events/        [existing: controlled integration only]
│   └── tests/av_scene/                   unit + adapter + integration checks
├── eval/av_scene/
│   ├── generate_fixture.py               truth-bearing synthetic fixtures
│   ├── score.py                          counting, attribution, stems, sync metrics
│   └── scenarios.json                   case definitions, no large media
└── quest/                               [existing: consume compatible hub results]
```

Use TOML configuration initially to reuse Python 3.12's `tomllib`; YAML remains an option if the team already standardizes it. Add package data for defaults and viewer assets to Hatch. Keep outputs and recordings under already ignored `hub/sessions/<run_id>/av/`; do not commit masks, generated videos, stems, model weights or tokens.

### 2.2 Protocol boundary

`Source` already carries label/confidence, optional bearing, state, level, `bbox`, `mask_rle`, visibility and observation time. `TargetUpdated` carries association confidence. These fields support an initial bridge without broadening the frozen schema.

- Use a run/epoch-scoped identity mapping. A model's `obj_id`, a separator slot and a public `source_id` are not interchangeable.
- Preserve normalized image bounding boxes and their source image orientation. Never send pixel coordinates into normalized fields.
- Define an internal RLE format with dimensions, order and foreground polarity. Verify the client's expected `mask_rle` encoding before publishing it. A string field alone is not an agreed codec.
- Keep the scene graph, score decomposition, unresolved candidate lists, event taxonomy and stem paths in the manifest. Do not insert extra fields into frozen wire messages.
- Keep model-job generation and the existing `target_id` and `selection_version` together in replay jobs. Discard stale completion after a new selection or epoch. The binary `AudioOut` header does not itself provide all job-version fields, so the publisher must enforce that check before sending.
- The protocol's two output channels represent playback layers or stereo; they do not represent an upper bound of two discovered sources. Mix the selected source and surrounding scene into the existing playback contract.
- Current tracker heuristics can merge same-label observations at nearby bearings and restrict bearingless births. Test and deliberately extend those policies before connecting multiple co-located or bearingless AV identities. Do not feed them new identities and assume they will preserve them.

Any required protocol changes are a separately reviewed integration decision. This specification proposes no schema mutation.

## 3. Processing architecture

```text
MP4 / recorded session / bounded live snapshot
   -> timestamped media + calibration + gap/visibility masks
   -> semantic events + speaker proposals + count hypotheses
   -> SAM 3 object tracks + face tracks
   -> ASD / spatial / audio-visual evidence
   -> provisional associations, including unmatched candidates
   -> per-candidate SAM Audio extraction
   -> identity, leakage, activity and residual validation
        -> bounded rediscovery / reassociation when useful
   -> accepted scene graph + stems + envelopes
   -> MP4 + interactive bundle + optional protocol bridge
```

Each stage accepts immutable artifacts and writes a versioned result. Model workers own their models. The orchestrator owns scheduling and source identity. CPU preprocessing and rendering may overlap with one GPU stage when memory permits; heavy GPU stages run serially by default.

Two operating modes use the same contracts:

1. **Recorded scene:** full timeline available, bidirectional tracks, whole-clip speaker identity, bounded global reassignment and export. This is the first complete product slice.
2. **Live assist + refined replay:** fast existing hub observations continue to drive the UI; slower AV evidence refines recent windows and selected-source replay. Publish result age and reject stale frame overlays. A delayed analysis result must not masquerade as an observation of the current frame.

## 4. Research-verified model contracts

These findings were checked against public upstream code/docs on 2026-10-25. URLs pointing to `main` are research references, not immutable dependency pins. Implementation must record exact repository commits, package versions and checkpoint revisions in preflight and every output manifest.

### 4.1 SAM 3 segmentation

The upstream README specifies Python 3.12+, PyTorch 2.7+ and CUDA 12.6+, with a current CUDA 12.8 installation example. The video interface uses `build_sam3_video_predictor`, `handle_request` and `handle_stream_request`; propagation is a streamed request. Current upstream also contains SAM 3.1, so a floating repository revision can change the available backends. [SAM 3 setup and examples](https://github.com/facebookresearch/sam3)

The base predictor dispatches session start/reset/close, prompts and propagation. Session close releases session state; model/worker lifetime is a separate responsibility. Export masks before resetting or closing. Namespace object IDs by backend, session and concept pass. [Predictor source](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_base_predictor.py)

Current `sam3_video_inference.py` resets inference state when adding a semantic prompt. Therefore, adding “person” at frame zero and again at an onset frame must be treated as separate proposal passes unless the pinned API proves otherwise. Merge exported tracklets explicitly; do not assume prompts accumulate. [Semantic prompt implementation](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py)

Proposed adapter behavior:

- Try frame zero plus event onset/midpoint only when discovery suggests missing instances.
- Run one concept/pass at a time, propagating both directions for recorded clips when supported.
- Deduplicate tracklets using temporal overlap, mask IoU and appearance, with explicit handling for occlusion and scene cuts.
- Preserve original-frame transforms after resize, crop, letterbox, rotation and mirror operations.
- Support `sam3_image` plus a separately validated tracking/interpolation adapter as the laptop baseline, and a pinned `sam3_video` backend for the full offline path. Tag native versus propagated masks and their age.

Cocktail's existing [architecture evidence](../ARCHITECTURE.md) reports a working SAM 3 image path and serious Windows video-tracker issues. Use that evidence to choose the first local backend; keep the full video-session implementation as a separately validated worker/runtime rather than silently substituting SAM 2 calls. These historical measurements were not rerun here.

### 4.2 SAM Audio extraction

Official SAM Audio supports text, visual and time-span prompts and publishes `small-tv`, `base-tv` and `large-tv` variants with improved target/visual prompting behavior. Use concise lowercase sound phrases. Read the processor's sampling rate; the documented tensor input rate is 48 kHz. Start the laptop profile with `small-tv` and expose larger checkpoints as quality profiles after measurement. [Official usage and variants](https://github.com/facebookresearch/sam-audio)

**Correct the polarity assumption:** current `Processor.mask_videos` multiplies video by the condition that the supplied mask equals zero. Internally retain `foreground=True`; convert to a background/removal mask for this helper. For RGB `T,C,H,W` frames, explicitly use a broadcast-compatible `T,1,H,W` mask. Verify preservation of a known foreground pixel and removal of a known background pixel, plus batch-length and shape checks. Merely inspecting the signature is insufficient. Tensor video inputs are mapped across audio feature positions uniformly, so the adapter must first align/crop the frame sequence to the exact audio window. [Processor source](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/processor.py)

The inspected model routes text, masked video and anchors into the separation computation together. Its current `separate` implementation returns per-item target/residual lists despite tensor annotations in the result dataclass and an older README save example. Normalize outputs in the adapter. A `separate_long` entry point was not verified in the inspected model implementation; implement Cocktail-owned chunking around verified `separate`. [Model source](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/model/model.py)

Require adapter smoke tests for text-only, visual-only, span-only and combined prompts. Log which prompts actually reached the model. Measure visual prompting separately: it activates a vision path absent from the repository's earlier text-only memory measurements.

### 4.3 Discovery and speaker timelines

PE-A-Frame and PE-AV exports are present under `core.audio_visual_encoder`. The official PE-A-Frame example returns temporal `spans`; PE-AV exposes modality-specific embeddings and demonstrates an audio/video dot product. Use the documented audio/video pair rather than assuming every embedding field shares one space. Do not hardcode an approximately 40 ms feature rate without inspecting the selected configuration. [Official examples](https://github.com/facebookresearch/perception_models#getting-started-with-pe-a-frame), [exports](https://github.com/facebookresearch/perception_models/blob/main/core/audio_visual_encoder/__init__.py)

Use the existing AST full class scores for candidate generation, followed by PE-A-Frame localization. Preserve fine labels before the live path's coarse mapping. A configurable taxonomy extends discovery with phrases such as “cat chirping”; an AudioSet tagger still has a bounded vocabulary. Include `unknown_event` proposals for persistent unexplained activity.

Community-1 accepts 16 kHz mono audio and exposes both `speaker_diarization` and `exclusive_speaker_diarization`. Use regular diarization for overlap; exclusive output is a separate view intended to simplify alignment with transcription. Its `num_speakers`/bounds concern speaker inventory, not a guarantee of correctly counting simultaneous talkers. [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1)

Record the pinned segmentation model's local speaker and simultaneous-speaker limits. Treat diarization as a proposal that can be revised using clean spans, visually active people and extracted stems. Do not copy numerical limits from a different pyannote checkpoint. A clip with three known identities is a different acceptance case from three people speaking at the same instant.

### 4.4 Active-speaker and spatial evidence

LR-ASD's upstream evaluation constructs 25 fps face tracks, 16 kHz audio/MFCCs and grayscale 112 by 112 model crops derived from larger face crops. Match that preprocessing, including timeline alignment. Its demonstrated score is thresholded around zero; do not label raw scores as calibrated probabilities. [LR-ASD evaluation source](https://github.com/Junhua-Liao/LR-ASD/blob/main/Columbia_test.py)

MediaPipe Face Landmarker can provide landmarks and facial transformation matrices. Use these for mouth-region tracking, visibility and pose-quality weighting. They do not replace calibrated camera-to-microphone geometry. [Official Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)

Pyroomacoustics SRP takes microphone positions, sample rate, FFT size and a source-count setting. `num_src=1` is a default, not an automatic multi-source discovery result. Keep multiple spatial peaks when appropriate and fuse them with source-specific evidence. [SRP-PHAT API](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.srp.html)

## 5. Time, audio and geometry contracts

### 5.1 One timeline with explicit transforms

- File mode retains original presentation timestamps and the original audio time base. Establish a documented run origin and map timestamps to integer microseconds plus sample/frame indices.
- Session/live mode retains `device_ts_us`, `epoch` and capture timestamps. Network receipt time measures latency, not source occurrence.
- Store `t_common = a * t_stream + b` for independently clocked inputs, with calibration confidence. A single offset is enough only when drift is negligible for the clip duration.
- Preserve variable frame rate. If a model needs constant-rate input, resample into an adapter view and record original-to-adapter frame mapping. Never calculate all original timestamps as `frame_index / nominal_fps`.
- Store audio discontinuities, padded silence and missing frames explicitly. Padding is missing-data handling, not evidence that the room was silent.
- Associate a mask with its actual frame timestamp; optical-flow propagation to a later frame creates a new estimate with its own confidence.

### 5.2 Preserve a high-rate master

Keep original multichannel audio for array processing and archival export. Derive a 16 kHz mono analysis view for existing hub models/diarization and a 48 kHz view for SAM Audio. Upsampling the existing 16 kHz ring does not restore discarded high frequencies.

For live refinement, add a bounded high-rate snapshot path before ingest downsampling, coordinated with the ingest owner. Keep the current 16 kHz consumers unchanged. Store channel order, microphone positions, preprocessing mode and channel synchronization evidence. “Two-channel WAV” can be stereo program material; enable physical DOA only for a calibrated synchronous capture array.

Canonical internal audio is float32 `(samples, channels)` with explicit rate and range; adapters transpose to their model layouts. Export sample counts, offsets and padding. Do not silently apply per-stem normalization before comparison or reconstruction.

### 5.3 AV history and capture

Add a byte- and duration-bounded video history for ASD and visual prompting while preserving newest-frame ingest. Decode on demand; cache compressed frames where useful. Record missing-frame intervals and avoid long interpolation through occlusion. Snapshot audio, video and pose from the same epoch/window without holding an ingest lock during GPU work.

Imported videos can be persisted for offline work. Live capture keeps the existing retention policy: bounded memory by default, explicit recording/export action for durable media. Cached embeddings, face crops, masks and voice profiles inherit the recording's lifetime.

For a later Quest bridge, Meta documents camera intrinsics, extrinsics and timestamps through its passthrough-camera API. Validate the project's installed SDK and timestamp conversion on-device before depending on those values; do not infer fresh device behavior from this research. [Official Unity camera documentation](https://developers.meta.com/vr/documentation/unity/unity-pca-documentation/)

### 5.4 Geometry

Use undistorted pixel coordinates and calibrated intrinsics. In camera coordinates, form a ray proportional to `K^-1 [u,v,1]`, then rotate it into the array/world coordinate system. Horizontal camera bearing is `atan2(ray_x, ray_z)` under an explicitly documented camera convention.

Translation affects a finite-distance point; it cannot be applied as a fixed angular offset to a ray without depth. If depth is unavailable, use far-field rotation and report the near-field uncertainty. Convert the repository's tracking convention and the DOA backend's axes explicitly, with left/right/front/back tests.

Distinguish the **wearer's head/camera pose**, which rotates the sensor frame, from the **observed speaker's face pose**, which primarily changes mouth visibility and acoustic directivity. Use a mouth landmark when reliable; use a larger region with lower confidence otherwise. Do not shift the measured source bearing merely because a person turns their face.

For overlapping sounds, use **W-Disjoint Orthogonality (WDO) raw-array masking** as an approximate dominance model, not a claim that speech bins are perfectly disjoint. On a shared STFT grid, form `W_k(t,f) = |S_hat_k(t,f)| / (sum_j |S_hat_j(t,f)| + epsilon)` from aligned mono stem/proposal magnitudes, then apply the same real non-negative weight to every original microphone spectrum: `X_m,k = W_k * X_m`. Inter-microphone phase belongs strictly to the original multichannel master `X_m`, never to a generated mono stem. Low-energy or poorly dominated bins supply no confident spatial evidence. [WDO/time-frequency masking research](https://doi.org/10.1109/TSP.2004.828896)

Preserve dominance weights in the GCC/SRP accumulation **after PHAT normalization**, or select bins explicitly: applying an identical soft mask to two channels and then normalizing their cross-spectrum can cancel that mask. Use weighted raw cross-phase, not a single mixture-level peak reused for all sources. Record mask provenance, effective bandwidth and retained peaks; validate phase preservation and source-specific delay recovery in fixtures.

### 5.5 Acoustic physics and psychoacoustic contracts

The following four invariants bind the proposed `locate/`, `who/`, `separate/`, `replay/`, `render/` and `eval/` adapters. Configuration values below are project defaults or physical approximations where stated; they are not new fields in the frozen wire protocol.

**1. Preserve acoustic time-of-flight (ToF).** Use dry-air `c_sound(T) ≈ 331.3 + 0.606*T_C` m/s, defaulting to 20 °C with an explicit “assumed” temperature flag. This gives about 343.4 m/s, 2.91 ms/m, and `tau_tof(r) = r/c_sound`: approximately 15 ms at 5 m or 29 ms at 10 m. Optical propagation is negligible at room distances; acoustic arrival follows the emitting motion. [Temperature approximation, NASA report, Eq. 3](https://ntrs.nasa.gov/api/citations/20190025444/downloads/20190025444.pdf)

After clock/codec offsets are accounted for, `who/asd.py` and onset-synchrony scoring must allow causal audio lag in `[0, 35] ms` without a timing penalty and penalize audio lead beyond clock/frame uncertainty. This is a default room-range prior, not permission to absorb arbitrary transport delay; farther sources require an explicit range-dependent extension. Align scoring windows or marginalize over allowed lags without shifting exported stem samples/timestamps to erase physical propagation. Keep speech articulatory-onset uncertainty separate from clock error. Section 10's playback gate includes ToF once, not as an additional allowance.

**2. Score 3D delay likelihoods, including ambiguity and calibration error.** A constant pairwise range difference defines a 3D hyperboloid (degenerating to a plane at zero delay); its far-field directions form a cone of confusion. Horizontal `atan2(ray_x, ray_z)` compared directly with `asin(c_sound*tau/d)` loses elevation (`cos(elevation)` compression) and front/back alternatives. `locate/av_spatial.py` must instead use:

```text
p1 = calibrated position of input channel 0; p2 = position of channel 1
b = p2 - p1; d = norm(b); b_hat = b/d
r_hat = R_cam_to_array * inverse(K) * [u,v,1] / norm(inverse(K) * [u,v,1])
tau_geom(r_hat) = d * dot(r_hat, b_hat) / c_sound
tau_exact(s) = (norm(s-p1) - norm(s-p2)) / c_sound
```

Rays point **toward the source**. For this channel ordering, positive delay means channel 0 arrives later, matching [`gcc_phat`](../hub/locate/tdoa.py) and [`frame_delays`](../hub/locate/pair.py). Compare predicted delay plus calibrated channel offset against measured TDOA likelihoods across calibrated pairs, or evaluate SRP-PHAT on the unit sphere `S^2`. Define camera/array axes and handedness before applying the transform. Known depth uses `s = R*(r*r_hat_camera) + t`; unknown depth retains angular uncertainty.

The existing pair model is `delay_amp_s * sin((theta - axis_deg - yaw)*pi/180) + offset_s`, with positive world bearing/yaw to the right. Its `delay_amp_s` is an empirical fitted amplitude. `axis_deg` is the sine phase: for a horizontal ideal model its equivalent baseline direction is proportional to `(cos(phi), 0, -sin(phi))`, i.e. baseline azimuth `phi + 90°`, not `phi`. Do not reinterpret the fit as independently measured geometry. Current `frame_delays` selects one GCC `argmax`; `PairLocator` retains an angular posterior/alternative but does not retain multiple raw delay peaks, and its pose path uses yaw only. Full 3D likelihoods are a proposed extension.

- **Parallax and curvature:** with unknown depth, require `r_min_m` (default 0.5 m) and `parallax_sigma` in radians of at least `atan(norm(t_cam_to_array)/r_min_m)` as a configured uncertainty floor; add its square to angular variance. That expression is a lateral-offset approximation, not an all-direction worst-case bound. For camera-to-source range at least `r_min_m > norm(t)`, use the stronger `asin(norm(t)/r_min_m)` bound for a conservative angular support; otherwise withhold a confident far-field match. Track curvature risk below the approximate Fraunhofer screen `r_FF = 2*D_aperture^2/lambda`, where `D_aperture` is the array aperture in metres and `lambda=c_sound/f`; use exact range differences or inflate delay uncertainty when the plane-wave approximation is inadequate. Do not confuse aperture with the code's fitted delay amplitude.
- **Aliasing:** use each pair's spacing to compute `f_alias,ij = c_sound/(2*d_ij)`. The requested array warning `f > c_sound/(2*d_min)` puts even the shortest pair above its half-wavelength criterion; longer pairs can be ambiguous at lower frequencies. `aliasing_multimodal_flag` is evaluated **per calibrated pair `(i, j)`**: it triggers as soon as the narrowband dominant frequency exceeds that pair's own limit `c_sound/(2*d_ij)`, so longer baselines keep multimodal peaks at their lower cutoff before `d_min` aliases. Expose a configurable WDO dominance floor (`wdo_min_weight`) that zeroes poorly dominated bins before post-PHAT weighting. Measure active-frame spectral flatness/effective bandwidth. For narrowband high-frequency content, retain multimodal GCC/SRP peaks and widen uncertainty rather than forcing `argmax`. A 3–5 kHz chirp or beep is an aliasing stress case only for an appropriate baseline, not automatically for every wearable pair. [Microphone-array spatial aliasing](https://vocal.com/echo-cancellation/spatial-aliasing-in-a-microphone-array/)
- **Head/visor/chassis diffraction:** allow an optional calibrated `baffle_scale = beta_baffle >= 1.0` effective-baseline approximation, default 1.0, plus frequency-dependent head-shadow confidence penalties or a measured array-response model. A scalar is a restricted engineering approximation, not the Woodworth law for arbitrary devices; Woodworth assumes spherical-head geometry and its errors depend on frequency/position. Do not multiply an already fitted `delay_amp_s` by an extra baffle factor: choose one calibrated prediction model and retain its residual uncertainty. Any headset profile requires device evidence before it is enabled. [Woodworth model analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3985894/)

**3. Prefer onset evidence in reverberation.** Under a diffuse-room, omnidirectional-source approximation, `d_c ≈ 0.057*sqrt(V/RT60)` metres, with `V` in m³ and `RT60` in seconds; source directivity adds a factor `sqrt(Q)`. At critical distance direct and reverberant energies are equal; beyond it the model predicts `DRR < 0 dB`. Values around 1–2 m are a useful indoor stress range, not a universal room constant. [Room-acoustics model](https://live.ocw.mit.edu/courses/21m-380-music-and-technology-recording-techniques-and-audio-production-fall-2016/d315c75efb01cb76ca897958570bab87_MIT21M_380F16_ses18_note.pdf)

Use computational **onset precedence (Haas-inspired weighting)**: `onset_flux_weight` increases for the first configurable 10–30 ms after positive broadband spectral-flux onsets and decreases for sustained vowels/reverberant tails. Estimate flux on raw-array/source-dominant evidence, not a potentially generated onset. This is a heuristic inspired by measured onset dominance, not proof that the first sampled peak is direct sound. Without a reliable onset, or when reflection peaks remain competitive, preserve ambiguity rather than attaching a wall reflection to a silent bystander. Store RT60/DRR estimates with confidence or `unknown`. [Reverberation and onset-dominance research](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805551/)

**4. Separate vocal radiation from geometric origin.** Speech is broadly omnidirectional at low frequencies (roughly below 500 Hz) and increasingly forward-directed above about 2 kHz. Use 6–10 dB high-band attenuation as a turned-away stress-test range, not a fixed cardioid law or a correction valid for every speaker/angle. [Measured voice directivity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3407162/)

For an associated speaker with observed `abs(face_yaw_deg) > 60°` or mouth occlusion: mark visual lip-sync ASD `unavailable`, widen high-frequency spatial uncertainty for possible reduced DRR, and freeze voice-profile learning. Do not move the geometric ray origin because of yaw. Gate **enrollment, return-window direct profile creation, history appends, learning-duration counters and periodic adaptation**, not just the final EMA write; recognition against an existing clean profile may continue. Bind eligibility to the audio interval and resolved identity/slot so a later clear face cannot approve previously contaminated audio. Existing [`VoiceMemory.reorder`](../hub/separate/voice_memory.py) accepts only voices and timestamps; its `_learning_blocked` handles uncertain identity, not pose. A future private quality input must combine both gates. Unknown pose is distinct from observed off-axis pose and requires an explicit learning policy.

## 6. Discovery, counting and anchors

Run broad tagging over overlapping windows, apply per-class thresholds and hysteresis, and merge compatible event spans. Store the raw label, normalized concept, audio prompt, possible visual nouns and alternatives. A doorbell may map to a door, a visible bell or an unseen device; absence of a door mask is not evidence that the sound was absent.

Keep separate outputs for total distinct speakers in a clip, active speakers at each time, active non-speech events and confirmed physical emitters. Maintain count hypotheses and confidence instead of forcing a fixed number from separator outputs. Cap candidate proposals by configured compute budget, not by a claim about the true number of sources.

For speaker activity `A_i`, high-confidence competing speech `A_j`, and interfering non-speech intervals `N`:

```text
positive_candidates(i) = A_i minus (union of other A_j and N)
negative_candidates(i) = union of other A_j minus A_i
```

Use half-open intervals, merge small gaps, trim uncertain boundaries and enforce minimum usable duration. Attach confidence and provenance to each anchor. A negative anchor says the target is absent; it must never cover a trusted positive interval. A positive anchor can contain other sounds in principle, but begin with high-purity evidence for identity selection.

**Fully overlapping speech may have no clean anchors.** Emit an empty anchor set with a reason, then use visible-instance prompts, earlier/later clean turns or candidate extraction. Tests must accept this valid state. Never manufacture solo spans to satisfy a fixture expectation.

Count refinement proposal:

1. Form initial speaker/event hypotheses from regular diarization and AST/PE-A-Frame.
2. Find high-quality solo regions and build source profiles when available.
3. Compare visible speaking candidates with the acoustic inventory.
4. Extract candidate stems with supported prompts and score duplicate content, identity and residual activity.
5. Add a source only when new evidence survives deduplication; merge only when evidence supports shared identity.
6. Stop at a small configured iteration/work budget; retain unresolved activity for inspection.

The existing two-voice separator is a useful proposal baseline. Repeatedly splitting a residual is an experiment with stop rules, not a proof of a third person. Review the repository's existing multi-speaker work before reimplementing it.

## 7. Association and refinement

### 7.1 Evidence ledger

For every candidate source/track pair, store valid intervals and quality-weighted evidence:

| Evidence | Useful for | Important distinction |
|---|---|---|
| Speech turn versus ASD timing | Linking solo turns to visible people | Marginalize over the causal ToF window from Section 5.5; off-axis/occluded lips make ASD unavailable. Simultaneous speakers can have identical activity timelines. |
| Candidate stem versus face synchrony | Testing a proposed speaker assignment | Separation artifacts can alter sync; keep original-mixture evidence too. |
| Audio/video embedding similarity | Ranking plausible sounding objects | Similar objects can share semantics; similarity alone does not establish instance causality. |
| Mouth/object motion | Supporting temporally plausible activity | A moving silent cat, chewing person or vibrating camera is a negative-control case. |
| DOA distribution versus camera ray | Compare 3D ray-predicted delays with calibrated pair/SRP likelihoods | Record `onset_flux_weight`, per-pair `aliasing_multimodal_flag`, `parallax_sigma` (radians), bandwidth, peaks and calibration/DRR quality; a scalar bearing delta is insufficient. |
| Voice identity/profile continuity | Linking speech across pauses and track breaks | Apply Section 5.5 learning eligibility to enrollment/history/adaptation; recognition does not authorize a profile update. |
| Track continuity and visibility | Preventing unnecessary identity switches | A visual track can disappear while the acoustic source persists. |

Start with an interpretable score using only available evidence. Normalize weights over valid signals, calibrate thresholds on held-out clips, and store the component values. A missing face, microphone array or embedding is “unavailable,” not a numerical zero pretending to be a negative observation.

Use gated bipartite assignment with explicit unmatched dummy candidates. Require both an absolute acceptance threshold and a margin over the next plausible candidate. Use time-local constraints and a continuity penalty; permit revisions when stronger evidence arrives. Whole-clip Hungarian assignment alone is insufficient when people enter, leave or swap positions.

Keep visibility and association status separate: `matched`, `ambiguous`, `unmatched`, or `rejected` describes a link; `visible`, `occluded`, `outside_view`, or `unknown` describes visibility. A failure to match is not proof of an off-screen source. The UI may display “unattributed sound” until off-screen evidence exists.

### 7.2 Two cats, one chirp

1. Detect a chirp window; enumerate cat tracks overlapping it.
2. Score each crop with corresponding audio, motion timing and any valid spatial evidence.
3. Compare against wrong-time audio, another cat crop and a silent window to detect semantic shortcuts.
4. Extract a candidate stem conditioned on each plausible mask, sequentially on the laptop.
5. Compare target activity, background leakage, cross-candidate duplication and independent spatial/sync evidence.
6. Accept a winner only when the margin clears calibration. Otherwise keep one acoustic chirp with two possible cats, retaining the stem for listening and later refinement.

This same path covers musical instruments, tools and machinery. Do not assume human ASD works on animal mouths; use an explicitly evaluated non-human motion/correspondence path.

### 7.3 Three people speaking at once

ASD can indicate that three faces appear to be speaking without assigning three acoustic identities. If diarization activity tracks are identical, temporal correlation cannot resolve their permutation. Use solo turns elsewhere, stem-conditioned synchrony, spatial evidence and identity continuity. If there are no such cues, preserve alternative assignments instead of randomly stabilizing one.

Define one-to-one matching for individual human-speaker hypotheses within a window, but allow a physical emitter such as a TV to carry multiple audible voice identities. Record `emitter_id` separately from `speaker_id` when that distinction matters.

### 7.4 Avoid circular confidence

Using a cat mask to extract a chirp and then calling that extracted chirp proof of the same cat is circular. Track evidence provenance. A refined stem may improve a hypothesis, but acceptance should also use independent timing/spatial cues, held-out windows or counterfactual controls. Present model-only associations as such.

## 8. Stem extraction and residual accounting

Use a 5-second context window with a 1-second overlap as an initial tuning point, not a model limit. Benchmark other lengths. Build every audio crop, video crop, mask crop and anchor list from the same global time interval. Convert anchors to chunk-local seconds, clip bounds, and preserve source identity through every overlap.

Align and trim returned samples, preserve a controlled gain convention, and use overlap weights that sum to one. Recheck identity across chunk boundaries; crossfading two different speakers produces a smooth but incorrect stem. Store per-chunk errors and support resume at the failed source/window.

Keep four different residual concepts in `hub/replay/residual.py`:

- **Per-query model residual:** the remainder predicted for one target extraction; it still contains other selected sources.
- **Raw waveform reconstruction remainder:** `r_wave(t) = x(t) - sum_k s_hat_k(t)` in a declared mono downmix, with alignment/gain metadata. Keep this difference as a diagnostic artifact, not the input to unexplained-event discovery. The current SAM Audio processor averages channels; preserve that rule and the multichannel master separately. Never implicitly broadcast mono stems across the array. [Processor channel handling](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/processor.py)
- **Spectral unexplained-event remainder:** phase-projected complex STFT residual or non-negative unexplained power, defined below. Use it for semantic/activity rediscovery and remainder/leakage scoring; label the projection method and its assumptions.
- **Unexplained acoustic events:** supported activity discovered from that spectral remainder, with confidence and source attribution still unresolved. It is not a new emitter merely because residual energy exists.

SAM Audio uses a flow-based generator and a `DACVAE` decoder; its waveform outputs must not be assumed sample-phase-locked to the input. [Generator/decoder source](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/model/model.py) For equal-power copies of one clean signal, raw difference power is `2*P*(1-cos(phi))`: quadrature or decorrelated phase gives about +3 dB relative to `P`, and opposite phase gives about +6 dB. These are illustrative phase-mismatch cases, not measured universal SAM Audio errors; zero mismatch gives zero remainder.

On a common, delay-aligned STFT grid, fit bounded non-negative gains `alpha_k` over short windows with smoothness/regularization. Either project candidate powers onto mixture phase with non-negative Wiener-style masks `M_k`, constrained by `sum_k M_k <= 1`, and compute `R_complex = X - sum_k M_k*X`; or use `P_unexplained = max(0, |X|^2 - sum_k alpha_k*|S_hat_k|^2)`. A complex short-window least-squares variant must constrain its transfer functions; never fit a free complex gain per bin that can explain away every observation. Store gain bounds, STFT settings, masks and projection error. An incomplete source inventory must retain unexplained mass; do not normalize known-source masks to sum to one by construction.

For detectors requiring a waveform, synthesize the projected residual, or use `sqrt(P_unexplained)` with mixture phase and documented overlap-add; mark the result as a projection proxy. Power subtraction assumes approximately additive/uncorrelated source powers, so test coherent-source cancellation and shared-bin interference separately. Phase robustness does not establish leakage accuracy: retain reference-stem/listening checks and a genuinely missing-source control. Do not sum per-query residuals or interpret high-energy remainder as proof of an off-screen object. Keep raw stems unchanged and evaluate any separately labeled mixture-consistent playback set for target purity.

Validate stems using reference separation scores where clean stems exist, interference/activity leakage, duplicate-stem similarity, speaker continuity and human listening. Generative quality and exact waveform reconstruction need separate reporting. Preserve faint valid sources; energy reduction alone is not a source-discovery objective.

## 9. Internal artifacts, caching and model lifecycle

Use typed dataclasses or Pydantic models with explicit schema versions. Do not put full waveforms or dense masks in JSON.

| Artifact | Required fields |
|---|---|
| `MediaIndex` | run/input hashes, original PTS, sample rates/layout, epoch/origin, stream transforms, calibration ID, gaps; `ambient_temperature_c` (default 20 with assumed/measured provenance), `r_min_m` (default 0.5) |
| `ArrayCalibration` (private) | mic positions/channel order, axes/units, camera intrinsics/distortion and camera-to-array transform; prediction model/version, channel offsets/noise, optional `baffle_scale` (default 1), bandwise scatterer uncertainty; fitted `PairCalibration` parameters kept distinct from physical geometry |
| `AcousticEvent` | event ID, label alternatives, start/end, activity score, audio prompt, visual noun candidates, provenance |
| `SpeakerTrack` | speaker ID, regular overlapping turns, optional identity profile, count evidence, anchor intervals/reasons |
| `MaskletTrack` | namespaced visual ID, concept, per-frame mask reference, image shape/transform, bbox, confidence, visibility |
| `Association` | source/visual IDs, validity interval, status, score components, runner-up margin, evidence dependencies; spatial peaks, `onset_flux_weight`, `aliasing_multimodal_flag`, `parallax_sigma`, bandwidth/DRR/RT60 confidence; causal lag prior, face quality and interval-level profile-learning eligibility |
| `SceneSource` | stable source ID, event/speaker/emitter links, lifecycle, visibility, current association revision |
| `StemResult` | source ID, model revision, prompt/anchor hashes, sample rate/count, offset, channels, raw/normalized output paths, errors; preserved arrival-time alignment and linked remainder IDs |
| `RemainderResult` (private) | kind: model / raw waveform diagnostic / phase-projected STFT / unexplained power; source stem IDs, mono downmix, sample/time base, STFT/gain/projection config, artifact path, diagnostic-versus-discovery use and proxy-waveform provenance |
| `RunManifest` | schema/config/code/model revisions, input hashes, stages, assets, timing, metrics, hardware and evidence labels; envelope filter/gate config, timing sign/reference, ToF/clock uncertainty and measured quantization budget |

These are internal artifacts and proposed private adapter inputs; do not add them to `hub/protocol/messages.py` or silently change current `PairCalibration`/`VoiceMemory` signatures. Cache keys include input content, preprocessing transforms, relevant config, checkpoint/code revisions, calibration and prompt/anchor changes, plus temperature, projection method and envelope/gating settings. Invalidate affected downstream stages when those inputs change. Use atomic completed-stage markers; a directory left by a crashed job is not a valid cache hit. Store source-specific artifacts independently so a mask correction need not rerun all discovery.

GPU lifecycle requirements:

- One GPU inference lease by default; no background job takes the GPU while a live session owns it.
- Report load time, preprocessing, inference, peak allocated/reserved memory, host memory and output duration separately.
- Release live references, close predictor sessions and terminate workers when necessary. `empty_cache()` does not free tensors still referenced by a model/session.
- Use validated autocast policies per adapter. Do not blindly convert every model's weights to half precision; existing SAM 3 measurements specifically distinguish autocast from weight conversion.
- Load only required SAM Audio components when the pinned model supports it. Visual prompting requires the visual path; text-only memory figures do not establish its footprint.
- On OOM, discard the incomplete stage, report the requested workload and try the configured smaller window/backend once. Do not loop indefinitely or hide a model/profile change.
- Expose an explicit budget/profile. `keep_in_vram` is an opt-in optimization after measuring the actual combination, not a guarantee based solely on GPU capacity.

## 10. Renderer and inspection player

In `hub/render/envelope.py`, apply an **ITU-R BS.1770 K-weighting biquad cascade strictly inside the visual-envelope branch**: shelving plus high-pass stages with coefficients validated for the actual sample rate. An explicitly selected IEC 61672 A-weighted alternative needs its own verified response. Neither path mutates exported WAV stems or the raw-array spatial branch. K-weighting reduces low-frequency influence and favors the chirp/consonant region, but is not a brick-wall sub-100 Hz rejection filter. Pair it with noise/activity gating to stop HVAC/handling leakage from activating silent people. [ITU-R BS.1770-5, Annex 1](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf)

Compute short-window RMS on that branch, apply a numerical floor, and map it to actual frame timestamps. Forty milliseconds is an envelope window, equal to one frame only at 25 fps. Label the weighted envelope explicitly; retain unweighted dBFS diagnostics. A weighted short-window RMS value is not LUFS. Full LUFS measurement requires the specified integration/gating process.

Pair approximately 30 ms attack / 180 ms release smoothing with a **reverberant-decay hysteresis gate**: separate activation and lower sustain thresholds, a fresh-activity/onset requirement to reopen, and a bounded tail hold after trusted source offset. A decaying room tail alone must not renew activity. Use RT60/DRR estimates when supported, otherwise a recorded fallback policy; do not suppress sustained speech or tonal sources merely because they lack repeated broadband onsets. Gate additionally on association and mask validity. Smooth visual fading is separate from the source's active flag and must not prolong “speaking” status. Test sub-80 Hz leakage against 2–6 kHz events at specified input levels; preserve quiet real sources with bounded normalization.

Render a consistent high-contrast palette, translucent fill, clear contours and compact labels. Provide a motion-reduced style. Encode identity through text/outline as well as color. The annotation must distinguish “speaking,” “possible match,” and “unattributed sound.” Do not display false precision such as an uncalibrated association percentage.

Proposed export bundle:

```text
hub/sessions/<run_id>/av/
  manifest.json
  annotated.mp4
  source_video.mp4
  mixture.wav
  stems/source_<id>.wav
  residual.wav                         projected playback proxy; method in manifest
  diagnostics/residual_waveform.wav    raw difference; never discovery input
  residuals/unexplained_power.npz      optional spectral discovery artifact
  masks/<track_id>.jsonl
  activity/<source_id>.json
  viewer.html
  assets/...
  metrics.json
```

The default is a self-contained **folder** with no CDN/runtime network dependencies. Serve it on localhost through the CLI; do not depend on browser `file://` fetch behavior. A single embedded HTML file is an optional small-clip export, not the default for large decoded audio buffers.

The player uses one WebAudio clock for all stems, shared scheduling offsets and gain nodes for solo/mute. Recreate buffer-source nodes after seeking and reschedule every stem together; use the video timeline as a synchronized display. Browser audio start requires the normal user playback gesture. The Web Audio specification defines scheduling against the audio context clock. [Web Audio scheduling](https://www.w3.org/TR/webaudio/#dom-audiobuffersourcenode-start)

Mute the video's embedded soundtrack when stems are active to avoid doubled audio. Provide an explicit original-mixture mode. Include play/pause, seeking, mute, additive solo, reset mix, source selection, residual toggle and evidence inspection. Clicking a mask resolves against that frame's masks in display coordinates; overlapping masks need an inspectable candidate choice.

Measure export sync and browser playback sync separately. Define `delta_audio_ms = t_audio_onset - t_visual_truth` and `delta_mask_ms = t_mask_onset - t_visual_truth`; **positive means late**. For a synchronized causal-onset fixture, both must lie in Cocktail's asymmetric **`[-15 ms, +45 ms]`** gate, expanded only by the measured visual quantization floor `q = 1000/(2*FPS)` ms: `[-15-q, +45+q]`. Define audio-to-mask skew as `delta_skew_ms = t_mask_onset - t_audio_onset` and bound it **symmetrically**: `abs(delta_skew_ms) <= 20 + q`, because audio playback and mask activation share the same physical acoustic ToF and a signed asymmetric window would mis-score a shared offset. This keeps opposite endpoint errors from hiding a large mismatch. For variable frame rate use half the actual local bracketing-frame interval. Record clock/annotation uncertainty separately; it is not an unlimited pass allowance.

This is a **project acceptance gate informed by psychoacoustic asymmetry**, not a verbatim ITU/EBU limit. ITU-R BT.1359-1 reports detectability around 45 ms audio lead / 125 ms lag using the opposite sign convention; EBU R37-2007 specifies 5 ms early / 15 ms late per stage and 40 ms early / 60 ms late end-to-end. [ITU-R BT.1359-1](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1359-1-199811-I!!PDF-E.pdf), [EBU R37-2007](https://tech.ebu.ch/docs/r/r037.pdf)

The `[0,35] ms` ToF prior consumes part of the +45 ms budget; do not add it again or advance exported stems to remove it. Measure decode/encode priming, resampler/filter group delay, envelope threshold crossing, scheduling and seek offsets within the remaining budget. A 30 ms attack constant is not permission to add 30 ms onset delay; compensate envelope-branch timestamps or revise gating while preserving captured audio timing. Test distance-dependent natural lag and non-causal audio lead separately. Sources beyond the configured ToF range need an explicitly declared evaluation profile, not a hidden timestamp correction.

## 11. Phase 0 findings and runtime plan

Read-only environment checks in this research session found:

| Check | Observed result | Interpretation |
|---|---|---|
| NVIDIA inventory | RTX 5070 Ti Laptop GPU; 12,227 MiB reported; driver 591.44 | Design around approximately 12 GB, with measured headroom. No inference benchmark run. |
| System Python | 3.14.8 | Not the hub runtime. |
| Existing `hub/.venv` | Python 3.12.15, PyTorch 2.11.0+cu128, transformers 5.20.2 | Inspect this environment and `uv.lock` before changing dependencies. |
| Model metadata in inspected hub environment | `sam3`, `sam-audio`, `torchcodec`, `pyannote.audio` not found in the checked distribution listing | Other task-specific environments were not inventoried; do not conclude the machine has no copies. |
| FFmpeg / ffprobe command lookup | Neither resolved on the inspected process PATH | Locate any existing installation or configure a project-local binary path before media tests. |
| C: free space | Approximately 613 GB decimal at inspection | Cache capacity is available; checkpoint access/downloads were not exercised. |
| Host RAM | CIM query denied | Measure through an available system interface before choosing CPU offload budgets. |
| HF authorization/checkpoints | Not checked or downloaded | Preflight must report cache/access status without printing tokens. |

The existing architecture records historical SAM measurements on this laptop. Treat them as prior team evidence, not fresh measurements and not a visual-prompt capacity guarantee. No model tests, physical-device changes, global installs or GPU jobs were performed for this specification.

Runtime plan: preserve the hub's Python 3.12 and CUDA 12.8 configuration. Resolve optional AV dependencies against the current lock in an isolated project environment, using the published TorchCodec/PyTorch/Python compatibility matrix. Test real audio and video decoding before model loading. Separate workers can share the same project runtime; use an explicitly pinned isolated worker environment only if dependencies or platform support require it, with the same artifact contracts. [TorchCodec installation/compatibility](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)

Preflight must report capabilities individually: CPU math, decoding, SAM 3 image, SAM 3 video, event discovery, diarization, SAM Audio text, SAM Audio visual, ASD and spatial calibration. Missing optional capabilities select a documented profile; missing required capabilities fail that requested stage clearly. Check gated access only with existing credentials and approved access; downloading checkpoints is a separate explicit action.

## 12. Build sequence, ownership and acceptance

The integration orchestrator owns `av_scene/`, internal shared contracts, this specification and coordination of root configuration. Respect PRD ownership for `ingest/`, `separate/`, `locate/`, `vision/`, `who/` and `replay/`. One editing agent per worktree; at most two delegated tasks active. A shared-interface change must have an agreed integration owner before parallel implementation.

### Phase 0 — Research and executable environment report

Deliver this design, exact upstream version candidates, a capability matrix, a small decode test and adapter contract probes. Keep research notes in this document and architecture links; do not overwrite agent instructions such as `CLAUDE.md` with application requirements.

Acceptance: distinguish documented contracts, local inspection, historical team evidence and untested hypotheses. Record actual model revisions and validated runtime combinations before inference implementation. This research change completes the documentation/inspection portion; model-loading and decoding probes remain implementation work.

### Phase 1 — Deterministic foundation and first inspectable fixture

Owner: integration engineer for `av_scene/`; ingest owner for `ingest/media_io.py`; perception engineer for interval math; render owner for a minimal viewer. Work serially or in separate worktrees after schema agreement.

Build the media index, typed artifacts, span math, gated matcher, mask codec and envelope logic. Produce a fixture with known audio stems, hand-authored masks and activity truth; render/export it before loading heavy models. This validates the whole output contract without confusing fixture success with learned perception.

Add these deterministic contracts under `hub/tests/av_scene/` before model integration (test descriptions only at this stage):

1. **3D pair-delay projection:** for nonzero elevations, compare camera-ray dot products with exact far-field limits, show the error of naive horizontal `atan2`/`asin` matching, and assert channel 0-later-positive sign. Swapping channels/baseline negates delay. Cover fitted `axis_deg` conversion, camera translation/depth, parallax floors and no double baffle correction.
2. **Narrowband multi-peak retention:** choose spacing and a tone above its pair aliasing cutoff; analytically enumerate admissible delay aliases, then ensure GCC/SRP evidence retains competing peaks and sets `aliasing_multimodal_flag` for that pair (a longer pair flags at its lower cutoff before a shorter one). Do not force source ID from a tied peak. Include a broadband control, temperature changes and a weighted-PHAT case proving soft dominance masks (with the `wdo_min_weight` floor) survive normalization.
3. **Phase-rotated residual:** use a real synthetic clean source and a unit-magnitude STFT phase rotation (conjugate symmetry for real synthesis), including polarity inversion. Verify that the raw difference can be large while a correctly gain-scaled projected/power remainder does not trigger an unexplained event. Add a genuinely omitted second source that must remain detectable; this rejects a projection that simply zeroes every bin. Assert mono channel shapes and unchanged raw stems.
4. **Perceptual envelope:** validate the K-weighting frequency response at the configured sample rate, then test equal-RMS sub-80 Hz rumble versus a 4 kHz chirp at documented levels. The combined weighted/activity gate rejects specified rumble leakage while detecting the chirp; test low-level chirps, sustained tones and an RT60 decay that cannot reopen the active flag. Assert exported WAV samples remain unchanged. Filtering alone is not the rejection guarantee.

Also exercise `[0,35] ms` causal lags, excessive audio lead, Section 10's signed mask/playback gate (`delta_audio_ms`/`delta_mask_ms` on `[-15-q, +45+q]`, `abs(delta_skew_ms) <= 20 + q`), and observed off-axis/occlusion blocking of every voice-profile learning path, including returning-speaker enrollment and history accumulation. Record exact fixture levels, STFT/filter/gate settings and expected values so tests are reproducible.

Acceptance: timeline and mask round trips; empty anchors for total overlap; no bystander highlights under known truth; unmatched candidates remain unmatched; seek/solo and mask onset satisfy Section 10's asymmetric gate with measured quantization. All deterministic contracts above must pass before learned inference claims. Label fixture results **mocked/synthetic**, not room/device validation.

### Phase 2 — First real model slice

Dependency: Phase 1. Owners: perception/vision and replay engineers in separate worktrees.

Implement preflight, mask polarity probe, AST discovery, one SAM 3 visual backend and SAM Audio small-tv extraction for a short recorded scene with one visible speaker and a silent bystander. Use configured text plus a real visual prompt. Save intermediate artifacts and a playable output.

Acceptance: tested model/runtime versions; foreground-preserving mask conversion; actual source-specific stem; bystander behavior measured; per-stage memory/latency; a listening review. Label results **replayed**, and model execution **hardware-tested on the named laptop** only when actually run.

### Phase 3 — Overlap, identity and same-class ambiguity

Dependency: Phase 2. Owners: `separate/diarization.py`, `who/`, `locate/av_spatial.py` and evaluation owners.

Add regular diarization, PE-A-Frame, LR-ASD, optional PE-AV, speaker profiles and the bounded refinement loop. Begin with one/two talkers; then test three-way overlap and two-cat cases. Add spatial evidence only with calibration. Compare every added model against the previous baseline and keep an ablation record.

Acceptance: end-to-end model metrics on held-out clips, explicit unknown states, deduplication and correct source continuity. A mocked similarity matrix is a matcher unit test, not proof that PE-AV resolves real cats or diarization recovers three simultaneous voices.

### Phase 4 — Long clips and full inspection output

Dependency: Phase 3. Owners: replay, render and orchestration.

Add chunked extraction, tracklet stitching, source/window cache resume, failure recovery, MP4 rendering, synchronized multi-stem playback and evidence inspection.

Acceptance: bounded memory as clip duration grows; stable identity through chunk boundaries; interrupted stages resume; no duplicate original audio; measured seek/playback sync and graceful handling of incomplete assets.

### Phase 5 — Live hub integration

Dependency: passing recorded path plus team agreement for ingest/runtime changes. Owners: ingest, events/track and device integration.

Add bounded raw-audio/video snapshots, GPU scheduling and the protocol bridge. Preserve event freshness and selection-version checks. Use replay results to refine scene identity without blocking capture or replacing an active selection with stale work. Review any client mask-codec or calibration additions as explicit protocol work.

Acceptance: all existing hub regression tests pass; disconnected/reconnected epochs cannot mix; canceled jobs do not publish; a slow model cannot grow an unbounded queue; real capture and display sync are measured on a booked device/GPU session.

### 12.1 Proposed CLI

Commands below are interface targets, not commands available today. Run from `hub/` to match repository conventions:

```sh
uv run python -m av_scene preflight --profile laptop
uv run python -m av_scene run --input scene.mp4 --output sessions/demo/av --profile laptop
uv run python -m av_scene run --session sessions/take01 --output sessions/take01/av --resume
uv run python -m av_scene inspect sessions/demo/av/manifest.json
uv run python -m av_scene serve sessions/demo/av --host 127.0.0.1
```

`--profile laptop` initially selects the measured image-based SAM 3 path, small-tv, one GPU job, one candidate at a time and optional PE stages. A separately validated `offline-video` profile selects native SAM 3 video tracking. Profiles list disabled evidence explicitly; they must never silently replace a requested model or change an evidence label.

## 13. Evaluation matrix and release gates

Maintain separate fixture-unit and learned end-to-end suites. Split recordings by scene/person rather than adjacent frames to avoid calibration leakage.

| Scenario | Required observation |
|---|---|
| One talker, two silent people | One acoustic identity; bystanders remain inactive. |
| Two alternating talkers | Stable identities across pauses and output-slot swaps. |
| Three talkers with solo turns plus overlap | Three identities; active-count timeline evaluated independently. |
| Three talkers always overlapping | Empty clean anchors allowed; measure count and assignment rather than forcing success. |
| Three talkers plus cat | Separate speech and chirp activity; no automatic assignment of residual speech to the cat. |
| Two cats, only one sounding | Correct attribution when evidence supports it; calibrated ambiguity otherwise. |
| Visible silent cat with off-screen chirp | Do not attach sound solely because a matching noun is visible. |
| Silent moving mouth / dubbed video / wrong-time audio | Counterfactual checks expose correspondence shortcuts. |
| TV speech and visible silent person | Acoustic voice versus physical emitter distinguished. |
| Person enters after frame zero | Onset prompting discovers and stitches the new track. |
| Occlusion, camera motion, scene cut | Track breaks and mask age handled explicitly. |
| Same direction, different speakers | Bearing duplicate heuristic does not erase distinct supported identities. |
| Mono, program stereo, calibrated array | Spatial capabilities enabled only for the valid input mode; distinct left/right-content fixture asserts declared mono downmix, exact output shapes and no implicit channel broadcasting. |
| Narrowband high-frequency source above array aliasing cutoff (`f > c_sound/(2*d_min)`) | Retain multimodal grating lobes; no confident wrong-cat/object assignment. Also test longer pairs at their lower individual cutoffs. |
| Reverberant room / reflective wall near silent bystander + speaker turned more than 60° off-axis | Onset weighting must reject the reflection-driven match or abstain; mark lip-ASD unavailable and block profile enrollment/history/adaptation without shifting the geometric ray origin. Measure against an unweighted baseline; do not assume precedence always removes reflections. |
| VFR, dropped frames, audio gap, epoch reset | No fabricated timing evidence or cross-epoch association. |
| Long clip, OOM, interrupted export | Bounded resources, explicit partial result and resumable stages. |

Report:

- Active-speaker count MAE and exact-count accuracy, stratified by overlap degree; total speaker inventory accuracy separately.
- Diarization error with overlap treatment and collar stated; identity switches per minute.
- Event detection precision/recall and onset/offset error by class.
- Association precision, coverage/abstention, wrong-object highlight time and calibration reliability. High precision obtained by rejecting everything is not success.
- Mask quality on labeled frames, track continuity and overlay age.
- SI-SDR improvement where reference stems are meaningful, interference leakage, duplicate-source extraction and listening scores; false unexplained-event rate under phase perturbations, split by remainder method, plus omitted-source recall.
- Spatial likelihood coverage and wrong-object attribution versus elevation, range, bandwidth and RT60; onset-weighting ablation; off-axis learning-block coverage and clean-profile drift.
- Signed audio/visual and mask/visual onset error under Section 10's `[-15 ms, +45 ms]` gate, `delta_skew_ms = t_mask_onset - t_audio_onset` under `abs(delta_skew_ms) <= 20 + q`, and browser seek error, all plus measured half-frame uncertainty `q`; ToF, clock uncertainty and processing delay reported separately. Also peak VRAM/RAM, load/inference/render time and real-time factor.

Suggested initial engineering gates, to be reviewed against a named held-out set: association precision at least 90% at at least 70% coverage on supported cases; silent-bystander highlight below 2% of annotated visible time; signed audio and mask onset error inside **`[-15 ms, +45 ms]`** and `abs(delta_skew_ms) <= 20 + q`, with only the measured `q = 1000/(2*FPS)` ms quantization allowance (half the local bracketing interval for VFR); no duplicate identities in deterministic fixtures. Preserve natural ToF within this budget and apply the corresponding lead penalty during association. Require the four Phase 1 physics fixtures, off-axis learning guards and signed timing cases to pass; real-room reflection/aliasing cases must meet attribution gates or remain explicitly ambiguous. These are proposed targets, not measured claims or a declaration of broadcast-standard compliance. Publish per-case results, including always-overlapping speech and same-class objects, rather than hiding them inside an average.

The first milestone is a short real clip that a teammate can open, inspect and listen to: source-specific masks, separate stems, a preserved residual, synchronized controls and traceable evidence. Expand to larger source counts, more object classes and live capture by adding measured evidence to that same end-to-end path.
