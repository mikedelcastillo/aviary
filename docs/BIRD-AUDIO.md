# Local bird-audio review prototype

The prototype can find short pieces of sound worth listening to, entirely on the development server using completed local clip copies. It does not yet establish that a sound is a bird, that a cockatiel is singing, or which individual made it. Nothing is connected to the running curator, event feed, or training export.

## What is implemented

`observatory/backend/audio_candidates.py` uses FFmpeg and NumPy. It measures energy changes and tonality in the available 700–3500 Hz band, merges nearby activity, adds 0.4 seconds of listening context, and keeps at most three non-overlapping candidates per clip. A sustained tone can qualify even when a whole clip is busy. Candidate ranking is a signal heuristic, not a model probability. Quiet chirps, lower-frequency sounds, or sounds outside the recorded bandwidth can be missed; speech, squeaks, whistles, and household machinery can qualify.

The CLI writes a separate private folder containing `report.json`, short PCM WAV clips, and `index.html` with audio players. Every source also gets a three-second middle-of-clip audit independent of the activity score, including sources with no candidates. The report records input hashes, codec/sample-rate evidence, segment boundaries, settings, processing times, and nullable review/species/individual fields. Inputs are opened read-only, a file changing during decode is rejected, and output must use a fresh directory. FFmpeg errors are summarized without including private paths. The code makes no network requests.

Run from `observatory/` in an environment with NumPy, FFmpeg, and ffprobe:

```bash
python -m backend.audio_candidates \
  --input /path/to/completed-local-copies \
  --output /path/to/new-private-listening-run \
  --source-rate 8000 --limit 9
```

Open the generated `index.html` locally to listen. The JSON and WAVs stay alongside it. No server or cloud upload is needed. Keep the folder private: a frame passing visual person screening does not prove its audio contains no conversation. The prototype does not redact speech.

## Camera bandwidth matters

The session's read-only probes found mono G.711 A-law at **8 kHz** from the current cameras. The recorder converts that to mono AAC at **16 kHz**. The higher container sample rate does not restore information above the original 4 kHz Nyquist limit. The CLI therefore records both container and known source rates and limits its band to the source's usable range.

For the pinned BirdNET v2.4 model, upstream specifies 48 kHz input, three-second windows, and a model-specific frequency range reaching 15 kHz. Resampling the camera files to 48 kHz satisfies the tensor shape; it cannot fill in missing high-frequency evidence. We need labeled recordings from these actual microphones before trusting species scores. [BirdNET v2.4 configuration](https://github.com/birdnet-team/BirdNET-Analyzer/blob/v1.5.1/birdnet_analyzer/config.py)

## BirdNET availability and the bounded trial

BirdNET-Analyzer is the maintained upstream analysis project. Its code is MIT licensed; its model weights use **CC BY-NC-SA 4.0**, with model-specific terms to review. The older BirdNET-Lite repository is archived and points users toward Analyzer. The model's noncommercial/share-alike terms are separate from the software license; the prototype does not bundle weights into this repository. [Analyzer and licensing](https://github.com/birdnet-team/BirdNET-Analyzer), [archived BirdNET-Lite](https://github.com/birdnet-team/BirdNET-Lite)

For a reproducible small trial, the experiment used the official v2.4 FP32 TFLite distribution rather than installing the full TensorFlow/Analyzer stack. The official ZIP is 76,822,925 bytes and its published MD5 is `c13f7fd28a5f7a3b092cd993087f93f7`; the download matched. The extracted `audio-model.tflite` is 51,726,412 bytes, SHA-256 `55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c`. Its matching English label list includes Cockatiel. All files remain in the isolated development experiment directory. [Official BirdNET v2.4 model distribution](https://zenodo.org/records/15050749)

`observatory/backend/audio_birdnet.py` is an optional local smoke-test runner. It takes the model and label paths explicitly, uses LiteRT on one CPU thread, checks the expected tensor and label dimensions, and reports top scores for three-second windows. It applies the v2.4 logit-to-sigmoid convention at sensitivity 1, records padded tails, and rejects subsecond tails. Scores remain unvalidated and are not calibrated probabilities of correctness. No geographic prior is used: a geographic filter could exclude captive species from an aviary. The runner has no individual-identity or song-versus-call classifier. [Upstream v2.4 model handling](https://github.com/birdnet-team/BirdNET-Analyzer/blob/v1.5.1/birdnet_analyzer/model.py)

After downloading the matching files separately into an experiment folder and installing `ai-edge-litert` plus NumPy in an isolated environment:

```bash
python -m backend.audio_birdnet \
  --input /path/to/completed-local-copies \
  --model /path/to/audio-model.tflite \
  --labels /path/to/labels/en_us.txt \
  --output /path/to/new-private-birdnet-smoke.json \
  --limit 3 --seconds 9 --source-rate 8000
```

This pins the v2.4 experiment; it is not a claim to benchmark the latest model family. Upstream now also describes newer model/runtime choices. LiteRT can run v2.4 without TensorFlow, making this small CPU experiment practical. [BirdNET library runtime options](https://github.com/birdnet-team/birdnet)

## Measured offline run on 2026-10-29

Nine completed clips were copied from the development server's app recordings, three per camera, selected without bird-audio scores from clips already given non-private visual statuses. Source files and services were not modified. This is a small convenience sample, not a balanced evaluation set.

| Measurement | Result |
| --- | --- |
| Decoded clips | 9 of 9 |
| Available audio | 179.840 seconds |
| Candidate segments | 15 |
| Candidate listening time | 16.548 seconds, about 9.2% of the audio |
| Independent audits | 9 × 3 seconds, separate from candidates |
| Decode, feature analysis, and source hashing | 2.688 seconds total |
| Elapsed through listening-WAV generation | 2.697 seconds |
| BirdNET smoke subset | First approximately 9 seconds of 3 copied clips; 9 windows |
| BirdNET model load | 0.072 seconds |
| BirdNET inference, one CPU thread | 0.608 seconds total, excluding decode and model load |

The BirdNET trial used LiteRT 2.3.0 and NumPy 2.5.4. Example **unvalidated** top scores were Northern Bobwhite 0.082, Ring-necked Pheasant 0.238, Yellow-browed Bulbul 0.373, and Trumpeter Swan 0.905. Human/engine classes also appeared. These are model outputs, not observations that those species were present. A high score alone is insufficient to populate the aviary feed. Cockatiel did not appear in the top five of these nine windows; without listened ground truth, that says nothing reliable about whether cockatiel calls were present or about model recall.

No person has labeled this sample for bird presence, singing, or identity. Precision, recall, missed-chirp rate, and identity accuracy are **not measured**. The processing times show that the offline pipeline runs cheaply on this host; they do not establish continuous-load performance or detection quality. The private `report.json`, `birdnet-smoke.json`, and listening WAVs are under the ignored audit artifacts, not source control.

## Review and integration proposal

Start with a small daily audio queue beside the image workflow: one short sound clip, a replay button, its synchronized video context, and **Bird sound / Other sound / Can’t tell**. A second optional question can distinguish **brief call/chirp / sustained song or whistling / uncertain**. Keep an independent audit slot and the original full clip available; otherwise false negatives remain invisible. Repeated views of the same event should share a source group so three camera microphones do not become three independent training examples.

Species and identity need stronger evidence than a sound score. An owner may recognize a characteristic song, but preserve that as a human judgment with uncertainty. A visible bird in the frame is not automatically the singer when other birds are nearby. Keep human labels, heuristic candidate scores, and BirdNET scores in separate fields. Uncertain or skipped audio must not silently become a negative example. Existing feed evidence should remain visual until an audio path is explicitly integrated and validated.

Before promoting audio labels, independently annotate complete clips from different times and cameras, including quiet intervals, faint calls, speech, water, cage movement, and machinery. Use whole recording periods as train/validation/test groups. Measure candidate recall for any bird vocalization and candidate precision to estimate listening burden; then evaluate species and call/song labels separately on the same held-out recordings. Individually identifying birds requires its own dataset and evaluation. Consider a wider-band local microphone only if the held-out test demonstrates that the present audio is insufficient; the present experiment does not justify a hardware purchase or an accuracy claim.

## Tests

Seven tests passed on the development server. Synthetic audio checks cover chirp-like bursts, merging/context, silence/rumble/stationary noise, source-bandwidth metadata, finite-input validation, independent audit sampling, source hashes, changed-file rejection, and private report generation. Mocked inference checks cover the explicit BirdNET tensor/label contract, sigmoid score order, padding, and short-tail handling. The real-model trial above separately verifies that the pinned model executes. No test contacted a camera, changed firmware, restarted a service, or sent audio to an external service.
