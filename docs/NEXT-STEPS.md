# Acceptance gates and next work

The current application is a development deployment. Component speedups and
successful short playback checks do not establish continuous recording, accurate
behavior recognition, or reliable identity. The measured results and unsuccessful
tests are recorded in [Observatory findings](OBSERVATORY-FINDINGS.md).

## Finish the video foundation first

1. **Measure source-to-screen freshness and recovery.** Keep one shared upstream
   connection per camera. Compare camera timestamps, relay output, recorder
   output and actual client frames, including natural reconnects. Exercise an
   already-open player through encoder and service restarts, lost input and
   malformed persisted state. A healthy process or recently written JPEG is
   insufficient evidence. Record missing footage explicitly.
2. **Evaluate a different camera input path.** On the development host, compare
   authenticated RTSP through the native relay with an FFmpeg packet-copy input
   and, where credentials and exact-model compatibility permit, authenticated
   Tapo transport. Replace one development source at a time with rollback;
   avoid adding concurrent upstream connections. Do not change production jobs
   or flash firmware as part of this experiment.
   The relay documents packet-copy input as an option for problematic RTSP
   sources; compatibility and improvement still require measurement on these
   cameras. [go2rtc RTSP guidance](https://github.com/AlexxIT/go2rtc/blob/v1.9.15/internal/rtsp/README.md),
   [FFmpeg source options](https://github.com/AlexxIT/go2rtc/blob/v1.9.15/internal/ffmpeg/README.md).
3. **Verify actual household devices.** Check Safari home-screen installation,
   audio activation, orientation and standby/resume on physical phones/tablets.
   Test TV remote navigation and Home separately from playback. OEM Home
   selection restrictions must not be reported as a successful launcher change.
4. **Measure six-camera capacity.** Report whole-stack anonymous memory, cache,
   CPU, GPU memory, encoding load, dropped frames and clip gaps. Include the
   local model service separately. Prefer video packet copy where input timing
   and browser codec compatibility genuinely pass the playback tests.

## Make daily watching useful

| Work | Existing foundation | Next acceptance gate |
| --- | --- | --- |
| Selected video Reels | Snapping playback, sound, favorites and seen timestamps | Human-reviewed clips with visible actions; reject duplicates, missing media and unsupported captions |
| Unseen-only daily edits | Completed recordings and view tracking | Preserve clips already watched; retain source/time provenance, audio continuity and an accessible original for every edit |
| Chronological bird feed | Indexed records and image/video entries | Explain gaps and uncertain identity; do not turn generated captions into established behavior |
| Better local visual judging | Local inference and a review queue; automatic publication disabled | A held-out, owner-reviewed video set containing negatives, people, feeder lookalikes and ambiguous actions; measure false claims and missed worthwhile clips |
| Bird sounds | Offline activity proposals and a local BirdNET trial | Owner listening labels for chirps, singing, household noise and silence; evaluate the actual narrow-band camera audio before publishing species claims |
| Conversational memory | Structured records and source retrieval | Explicit identity uncertainty, linked evidence, corrections, time ranges and cross-session follow-ups; counts and durations computed outside the language model |

## Improve observation through small daily reviews

The in-app tagging workflow offers full frames, manual boxes, uncertainty, undo
and a small daily goal. It samples beyond the current detector and rejects near
duplicates. See [daily tagging](DAILY-TAGGING.md). The next useful dataset is
short video tracks with visible identity changes and occlusion, alongside
detector-independent negative frames. More copies of an almost identical
photograph do not solve the missing examples.

Split evaluation by capture group and date before training. Report per-bird and
day/infrared performance, including unknown birds. Keep a separate human-labeled
video set for identity switches and feeding/drinking actions; a detection box or
feeder-zone visit alone cannot establish those behaviors.

## Build calibrated spatial behavior in stages

1. Validate printed wall and cage marker geometry in daylight and infrared at
   the intended camera distances. Record the marker bundle and physical size.
2. Calibrate camera intrinsics and estimate camera/cage pose from enough visible
   marker corners. Detect relocation and invalidate stale pose rather than
   stretching old overlays onto a moved scene.
3. Tag feeder, water, floor and perch geometry in the appropriate cage frame.
   Project zones only while pose confidence and visibility are sufficient.
4. Synchronize camera observations and evaluate multi-view bird tracks with
   measured reprojection error, identity switches and explicit coverage gaps.
   Only then produce metric trajectories and daily spatial summaries.

Current image-reference guards are 2D. Printable markers and a proposed geometry
format are preparation, not working 3D tracking. See
[movable-camera calibration](MOVABLE-CAMERA-CALIBRATION.md) and
[immutable zone references](CAMERA-ZONES.md).
