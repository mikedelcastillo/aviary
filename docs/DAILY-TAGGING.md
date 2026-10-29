# Daily five

Daily five lives in the main observatory app. It turns a small number of distinct full frames into explicit owner-reviewed examples for the existing annotation and training workflow. It does not train or replace a model automatically. More recordings alone do not establish better detection: the useful additions are missed birds, false positives, different poses and lighting, and genuinely empty frames that someone checked.

## Reviewing on a phone

1. Inspect the whole still frame, including outside the dashed detector suggestions. Zoom cycles through 1×, 2×, and 3×; drag the enlarged frame to look around. Nearby video helps identify a bird, but the boxes must describe the still frame.
2. Use **Add a missed bird**, then tap two opposite corners or drag around the entire visible bird. Select a box and use **Redraw** to fix its bounds, or **Remove** for a false positive. Drawing does not require a detection. Zoom first when a bird is small.
3. Label every bird. Choosing a label advances to the next unlabeled box. **unknown bird** means a bird is present but its identity is unknown. Infrared uses species labels; phone collection images use the original archive roster. The lighting switch clears labels so a color identity cannot silently become an infrared species annotation.
4. Check the whole-frame confirmation and save. An empty frame requires removing false suggestions and explicitly confirming that no birds are visible. **Can’t tell** counts toward today's five but is excluded from training. **Set this one aside** also stays out of training and does not count toward the five; it defers that item instead of repeatedly putting it in front of you.
5. Stop at five. More reviews are optional and explicitly requested with **Five more, if you feel like it**. **Undo last answer** reopens the last submitted frame with the edits restored while this review session remains open.

The count uses the configured app timezone and all active owner reviews, including uncertain answers. Changing or undoing an answer updates that count. Submission conflicts return an error instead of silently overwriting someone else's answer. A frame that failed to load cannot be confirmed empty in the UI.

## What reaches the queue

The curator offers its middle full frame even when the bird model returns zero boxes. Person-screened clips are the source; the detector is allowed to suggest boxes but never writes training truth. Roughly one quarter of clip IDs are marked for audit independently of bird scores, and all zero-detection frames are also audit candidates. The daily selection reserves an available audit slot, balances camera and sampling reason, and avoids repeating a capture window within a batch.

The sampler compares a whole-frame perceptual hash plus 12 image-tile hashes, separately for color and infrared. It keeps at most one frame per camera per ten-minute capture window, checks the last 300 retained samples from that camera for near repeats, and caps the pending queue at 180 by default (`learning_queue_limit`). Budget, window, and duplicate checks share a write transaction so a backfill and curator cannot race past the limits. A retained still and its video context are copied into the app runtime so rolling recording retention cannot break an offered card. Reviewed media is retained for export and undo; the pending cap is not a disk-retention policy.

This is a targeted review queue, not an unbiased estimate of bird prevalence or detector accuracy. The middle-frame rule, availability of recorded clips, person screening, camera activity, near-duplicate suppression, queue capacity, and owner decisions all affect selection. Small birds, motion blur, exposure changes, or a moved camera can defeat a perceptual-hash heuristic. An audit item means independent of a bird detection, not independent of those other filters. Model performance has not been shown to improve until a new model is trained and evaluated.

## Bringing in an old image collection

The optional importer reads a **local copy** with the existing layout: `tapo/day`, `tapo/ir`, and `phone`. It does not mount or write to remote originals. Run it on the development server, from the app's backend working directory, using the environment containing OpenCV and NumPy:

```bash
AVIARY_HOME_CONFIG=/path/to/private/config.json \
  /path/to/ml-environment/bin/python -m backend.learning_import \
  --source /path/to/copied-annotation/raw --limit 15 --max-scan 500 --seed 7
```

The paths above are placeholders. The configured runtime is the destination. It must be separate from the source tree. The importer shuffles filenames with the chosen seed, reads no detector scores, offers full frames without proposal boxes, and uses the same queue budget and deduper. It copies accepted frames as JPEGs; originals and every JSON/TXT sidecar remain untouched. A source-content SHA-256 prevents repeated copies of the same file from becoming new tasks. Links outside the copied tree are skipped.

Completed canonical JSON annotations, including reviewed empty images, are skipped by default. Malformed canonical sidecars are also skipped. `--include-reviewed` allows an explicit fresh review of completed annotations. An old `.txt` detection file is never treated as a human label. The summary reports added, known, duplicate, reviewed, invalid, and other skipped counts. Only JPG/JPEG/PNG files directly in the three category directories and with filenames accepted by the old annotator are considered. Change the seed or increase the scan limit if the sampled files mostly duplicate earlier selections.

An imported still has no video button. Its category and original basename survive export; its source hash and capture-time basis are in provenance. Recognized Tapo filenames retain their original camera/date/ten-minute split grouping. For other names, displayed time is the file modification time, **not a claimed capture time**; their temporal relationships cannot be reconstructed. The old corpus was often collected only after a detector fired. Sampling it without detection scores cannot recover frames that were never collected; keep detector-independent sampling of new recordings enabled.

## Export, retrain, and evaluate

**Export for the annotation project** produces only active, completely confirmed annotated or empty frames. It includes the images, canonical `{v: 1, boxed, boxes}` JSON, YOLO TXT using the configured **original global roster indices**, and a manifest. Missing or conflicting roster indices prevent export. Unknown labels, out-of-frame geometry, duplicate box IDs, and partial confirmed frames are rejected. Uncertain and deferred answers are retained in the database but never exported as negatives. Empty TXT files in this export mean explicitly reviewed empty frames.

Extract the ZIP into a new annotation data root and point the existing annotator's `AVIARY_DATA_ROOT` to it. Use the same original roster with `AVIARY_ROSTER`. Do not overwrite the original collection. Imported frames preserve original stems (with a `.jpg` image extension); recording samples use `learn_YYYYMMDD_BBB_24hex`, where the UTC day and ten-minute bucket keep simultaneous camera views in one training split. Canonical boxes use normalized center, width, and height, matching the old annotator. The manifest preserves source group, camera, capture time, clip ID or imported-source hash, frame offset, review time, and answer ID. Two imported images that would collide at the same export name cause an explicit export error.

Before combining this export with an existing dataset, deduplicate imported originals using source name/hash and keep all related capture windows together. Preserve a fixed held-out set of independently reviewed complete frames from unseen recording periods. Never train on that set or on neighboring frames of its clips. Do not treat different camera views of the same event as independent validation examples. Old unknown phone-photo bursts need manual grouping because timestamps may be missing.

Compare the current model with a candidate on that same held-out set: detection recall for missed birds, precision/false positives, localization, and identity/species confusion, broken down by camera and day/IR. Report the number of frames and birds per slice, especially rare classes. Include empty frames and scenes that the old detector missed; a benchmark consisting only of its successful detections cannot measure those misses. Keep the existing model active until the measured tradeoffs justify replacing it. A more pleasant labeling workflow and more varied examples are useful changes; an accuracy gain remains an empirical question.

## Verification

`observatory/tests/test_learning.py` covers complete-frame and lighting validation, canonical/YOLO export, review provenance, undo ownership, uncertainty versus true negatives, daily-goal limits, imported-phone labels, and missing video. The sampler tests cover independent zero-detection admission, full-frame/tile deduplication, concurrent offers, and expired context. Importer tests verify unchanged source hashes, ignored detector sidecars, skipped completed/corrupt annotations, duplicate admission, and source-tree boundaries. Run these tests and the web typecheck/build on the development server, without changing the production collector or archive host.
