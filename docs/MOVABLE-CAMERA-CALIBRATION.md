# Movable cameras, cages, and printable markers

Design requirements recorded 2026-10-29. Marker-based 3D reconstruction is not
yet deployed. The current view-alignment guard is an interim 2D check.

## Physical arrangement

The intended installation has six cameras. Two PTZ cameras are inside two
different cages. Outside cameras include both PTZ and cameras without motors.
All are potentially movable tabletop devices and can be knocked over. Cages
also move. "Non-PTZ" describes a motor capability, never a permanent pose.

Expected marker distances are approximately 1–4 m. The room is L-shaped; some
cameras do not see any cage. Use room-relative zones for those views, and link
the calibration graph through shared wall references or measured transforms.
Disconnected views remain in separate frames until such a link is established.
An unseen corner is a coverage gap, not evidence of a continuous bird trajectory.

Store separate facts for camera capability, attachment, and calibration:

- Capability: pan/tilt, zoom, audio, stream formats, pose telemetry availability.
- Attachment: room/tabletop, cage frame, or unknown; an explicit rigid mount is
  stronger evidence than a camera merely resting inside a cage.
- Pose: reference frame, transform, intrinsics version, timestamp, observed tag
  IDs, residual error, and validity. A PTZ encoder is not a world pose.

## Coordinate frames

Use a room frame established by stationary wall markers and a distinct local
frame for each cage. Feeders, water, and perches belong to the cage frame when
they move with it. Each camera has a mount frame and an optical frame; pan,
tilt, zoom, mounting changes, and physical relocation affect its optical pose.

Internal cameras see inward-facing cage references. Outside cameras observe
the cage marker bundle and wall references. Where these observations connect,
the system can transform cage-relative observations into room coordinates.
Internal cameras do not need to see the wall continuously. If no measured
connection exists, retain cage-relative results and report the room pose as
unknown; never manufacture a connection between coordinate frames.

Two cameras in different cages are not automatically a stereo pair. Bird 3D
triangulation needs overlapping views of the same bird, calibrated intrinsics
and extrinsics, useful angular separation, and measured time alignment.
Unknown stream delay and rolling shutter limit moving-bird accuracy.

## Ball-shaped cage markers

A plain sphere provides no unique orientation. Colored spheres also lose
useful color separation under infrared night vision. Standard square tag
detectors assume planar tag geometry, so wrapping a square code around a
sphere is not interchangeable with a flat tag.

The proposed cage marker is a rounded, ball-like body with several flat coded
faces. Give every face a distinct AprilTag ID and export its exact corner
coordinates in the body frame. Detected corners from multiple faces can enter
a joint pose solve. A camera does not need to see every face at once.

Provide flat wall plates and inward-facing cage plates as companion designs.
Prefer multiple rigidly mounted bundles on separated cage corners where
occlusion warrants it. A freely rotating ball or a loose clip cannot locate
the cage reliably. A mount needs a keyed orientation and a measured transform.

The print export should include two material bodies for black and white,
the dictionary and IDs, units, marker border dimensions, per-face transforms,
and a calibration manifest. Keep code faces flat, high contrast and matte.
Filament contrast must be checked with the actual infrared illuminators; a
good-looking daytime print is not evidence of night readability. Print size
is configurable and must be chosen from camera distance and pixel coverage.

At longer distances, marker detection should use occasional native-resolution
frames or native-resolution regions, even when the normal mobile video is
720p. Downscaling every source frame before detecting small distant tags can
remove the code detail needed for reliable calibration. Benchmark this separate
low-rate path; do not add another continuous full-resolution decoder per camera.

This design is consistent with multi-face fiducial research and tools such as
[AprilCube](https://github.com/younghyopark/aprilcube) and
[quARtet](https://github.com/Wa-Araki/quARtet-marker). Those projects are useful
references, not evidence that this installation has been calibrated.

## Relocation and recovery

1. Monitor visible marker residuals, global image motion, lost references, PTZ
   movement, zoom changes, and stream restart events.
2. Mark affected poses invalid immediately when inconsistent. Do not reuse a
   stored pose merely because the picture stops moving or the motor reports
   the same angles.
3. Continue video and recording. Suspend automatic aiming, triangulation,
   cross-camera identity transfer, and location-based visit accounting that
   depends on the invalid transform. Manual PTZ controls can remain available.
4. Recover from identified tags with sufficient known geometry and image
   coverage; reject poor residuals and ambiguous pose solutions. Require
   repeated consistent estimates before resuming.
5. If all references are obscured, show "Position unknown" and ask for a
   visible reference or manual recalibration. Never silently accept a new
   camera orientation as the old one.

## Interim software guard

Zone visit accounting now requires alignment to a stored reference image,
matching static features distributed across the view and near the zone, plus
three consecutive confirmations. Bird boxes are excluded from matching.
Missing references, featureless scenes and displaced views fail closed. PTZ
motor-angle checks remain an additional condition.

This is deliberately conservative and can pause under night-mode changes,
occlusion, or repetitive cage bars. It is not metric 3D pose estimation and
does not establish that an object beside a bird is food or water. Existing
zones without reference images need re-creation. A matched view is evidence
of image alignment, not a guarantee that every cage object stayed put.

## Validation before enabling 3D features

Validate printed geometry and tag IDs independently of the renderer; test
actual camera imagery at several distances, oblique angles, day/IR lighting,
partial occlusion and motion blur. Move and rotate cameras and cages separately,
including an obscured-reference condition. Measure false recovery, reprojection
error, pose repeatability and detection time. Establish a synchronized,
manually annotated multi-camera video set before making bird identity or
feeding/drinking accuracy claims.

## Projected overlays and daily movement history

Feeder, water, floor and perch boundaries live in the appropriate cage or room
frame. Project them through the latest valid camera transform, intrinsics and
lens-distortion model into the video. Account for stream crop, letterboxing,
resolution and display timestamps. Overlay metadata must correspond to the
displayed video time, not the newest inference frame: HLS video is delayed.

Track observations need timestamps, camera/track ID, identity probabilities,
2D geometry, optional 3D location and uncertainty, plus the pose version used.
Use individual identity only when supported; preserve anonymous track segments
and uncertain cross-camera associations. Summaries can report observed floor
time, distance during tracked coverage, zone proximity and uncovered intervals.
They cannot infer inactivity from missing video or consumption from proximity.

Sources: [AprilTag](https://github.com/AprilRobotics/apriltag),
[OpenCV marker detection](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html),
[multi-view marker pose research](https://arxiv.org/abs/2103.09141).
