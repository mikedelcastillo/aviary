# Tapo control and firmware notes

Checked 2026-10-29. The session's read-only inventory identified Tapo C216 hardware v1 and Tapo C320WS hardware v2. Camera addresses, credentials, and device identifiers belong in private runtime configuration. No camera movement, calibration, reboot, firmware change, or replacement-firmware installation was performed while implementing the Stop fix.

## Local control

The observatory uses the existing local ONVIF camera account for bounded directional steps, saved viewpoints, and Stop. tolpo's protocol notes distinguish standard RTSP/ONVIF camera-account access from the proprietary Tapo video channel and other authenticated local transports. Their documented observations are useful protocol references, but do not establish that every Tapo model or firmware supports the same calls. [tolpo protocol notes](https://github.com/signerless/tolpo/blob/main/docs/protocols.md)

The important control principle from tolpo is that stopping a camera should not wait for a capability read. Its `ptz_stop` path sends the stop command without first fetching the component list. The observatory applies that principle to ONVIF: Stop bypasses the ordinary per-camera move lock and does not request current position or presets. It uses a cached profile token, or obtains the token if this process has not discovered one yet. [tolpo camera-system implementation](https://github.com/signerless/tolpo/blob/main/src/tolpo/areas/camera_system.py#L196)

A lock bypass alone was insufficient. An older request could be waiting for `GetStatus`; once Stop completed, the old status response could arrive and cause a new `AbsoluteMove`. The controller now increments a per-camera generation immediately when Stop arrives. Every non-Stop request records its generation before entering the queue and checks it again after profile/status/preset work and before issuing a movement command. Older queued or preparing work returns `cancelled: true`. A failed Stop still invalidates old work. A new deliberate move after Stop gets the new generation; other cameras are unaffected.

This cancels commands still waiting or preparing in this controller. It cannot retract an HTTP movement command already sent to a camera, guarantee network delivery, or prove that a camera motor stopped. The actual Stop request remains necessary, and a rejected or timed-out Stop remains an error. The generation and queue are in memory, so PTZ requests must continue to use a single controller process; multiple independent API workers would require shared cancellation coordination.

On the phone, Stop stays available while another movement request is busy. It has its own pending state, so completing an old request cannot re-enable controls during a pending Stop or replace its error. Changing cameras clears stale shortcuts and ignores responses from the previous selection. Selecting a camera known to have no PTZ motor closes the movement drawer, and its movement button is disabled.

## Official firmware findings

Firmware listings are specific to hardware version and region. The following is public release-note evidence, not a claim that a particular release is installed or available through every regional app rollout:

| Device | Release listed on the checked vendor page | Release-note meaning |
| --- | --- | --- |
| C216 v1, EU and KR entries | 1.3.2 Build 260820, published 2026-09-03 | Improved device connection stability. [TP-Link Korea C216 v1](https://www.tp-link.com/kr/support/download/tapo-c216/v1/#Firmware) |
| C320WS v2, EU entry | 1.6.6 Build 260917, published 2026-10-09 | Improved device security. This is listed above 1.6.5 Build 260806, published 2026-08-12, whose notes mention security and stability. [TP-Link C320WS v2](https://www.tp-link.com/en/support/download/tapo-c320ws/v2/#Firmware) |

Thus 1.6.5 should not be described as the latest public C320WS v2 EU release as of this check. An installed version or an app offering can legitimately differ. The C216 connection-stability note is a reason to investigate a matching official update if connection trouble persists; it is not evidence that firmware caused a specific dropout or that updating will fix it. No firmware update is part of this code change.

## Replacement firmware

The OpenIPC builder identifies itself as experimental and lists support by specific device and hardware combination. Its checked device table lists several Tapo variants, but does not establish C216 v1 or C320WS v2 support. Support for another Tapo model, or a similar chipset, is not evidence of a working image for these devices. This is a limitation of the evidence reviewed, not a claim that a future port is impossible. No replacement firmware is proposed or installed here. [OpenIPC builder device table](https://github.com/OpenIPC/builder)

## Verification without moving hardware

`observatory/tests/test_ptz.py` replaces all ONVIF calls with async mocks. It checks Stop while a status read is delayed, cancellation of an already queued anchor, both home-command paths after delayed preset discovery, first-profile discovery, failure of Stop, and independence between cameras. The six cases passed on the development server, alongside the web TypeScript check and production build. No test opens a connection to a real camera.
