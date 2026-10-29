# Aviary for Android TV

The native TV client and Home launcher live in `observatory/android-tv`. They
support Android TV 8.0 or later, including remote-only use. The TV connects
directly to the home server over Wi-Fi or Ethernet. It does not need Tailscale.

## Connect and watch

1. Install the debug APK using the TV's normal sideloading workflow, or use
   `adb install -r aviary-tv-debug.apk` after the owner enables and authorizes
   debugging on that TV. The application ID is `org.aviary.tv`.
2. Open **Aviary**. It automatically finds `_aviary._tcp` advertisements on the
   home network and verifies `/api/discovery` before accepting a server.
   Enter the home passcode. **Scan again** retries discovery; **Enter address**
   exposes the manual URL field only when needed. Deployment details are never
   compiled into the APK.
3. Use the D-pad to select a camera. OK opens full screen; Back returns to the
   grid with the same camera selected. The wall fills its tiles with proportional
   center cropping, without stretching or scrolling. Full screen shows the whole
   frame. There are no permanent headings or camera-name bands.
4. Home starts muted on every visit, including Home pressed while already on
   the wall. **Sound on** enables only the selected camera for that visit;
   returning to the wall mutes it again without changing TV/system volume.
5. Open **Replays** for approved Highlights or all Recordings. OK plays a clip;
   **Back 10s** and **Forward 10s** seek. Remote media keys also pause and seek.
   **Older** and **Newer** page through the archive. Back returns to the list,
   then the live grid.

Press Up from a top-row camera, or the remote Menu key, to reveal Replays,
Sound and connection controls. They disappear after six seconds. The Home
screen has a device-local clock and a horizontal row of chosen apps below
the camera wall. Left/right selects an app, OK opens it, and Up returns to a
camera. Video players release when another app opens and return with Home.
There are no recommendation feeds or advertisements.

With three cameras, one larger view occupies the left two-thirds and two views
stack on the right. All three tile shapes match, so each feed receives the same
proportional crop; there is no empty fourth cell. Other camera counts use a filled
grid. The wall uses Media3 `ZOOM`; full screen and replays retain `FIT`. On the
tested 16:9 TV with three 16:9 feeds, the wall crops about 18% of the source width
in total. Open a camera to inspect its complete edges. The app strip is 104 dp
high and remains separate from video. Charcoal surfaces, warm text, sage focus
accents and bundled Albert Sans match the web client; no font network request is
required. The font's SIL Open Font License is included in the Android project.

Pinned apps appear first; **Apps** and system **Settings** stay at the right end
of the horizontal row. Native TV banners appear without outlined cards, with a
centered icon/name fallback. Focus uses a modest scale and brightness change.

Hold OK on a pinned app to enter arrange mode, then release. Press OK once to
pick it up, use Left/Right to move it, and press OK again to drop it. When no
app is picked up, Left/Right chooses another pin. Back drops the app and finishes
arranging, saving the order. A Move badge identifies the picked app; short remote
hints appear only during arrange mode. Up/Down stays in the row while arranging.
Home also exits arrange mode and returns to the muted camera wall.

The **Apps** tile opens the full installed-app drawer. **Choose apps for the
row** shows checkboxes; OK toggles an app, Right focuses Save, and Up returns
to the list. Save to show or hide apps. **Arrange the row** is another entry to
the same direct arrange mode. Reordering and visibility changes update only
the app strip: existing camera players, selected camera and sound choice remain
unchanged. Saving the picker returns focus to Apps.

The chosen order persists on this TV. Missing or uninstalled apps are skipped.
Apps and system Settings remain accessible even when every optional pin is
hidden, the server is unreachable, or sign-in has expired. These choices never
uninstall applications.

## Use Aviary as Home

After verifying the APK on the intended TV, record its current default Home:

```sh
adb -s TV_SERIAL shell cmd package resolve-activity --brief \
  -a android.intent.action.MAIN -c android.intent.category.HOME
```

With the owner's authorization, set Aviary using the supported Android package
command, or the TV's default-app chooser if that command is unavailable:

```sh
adb -s TV_SERIAL shell cmd package set-home-activity org.aviary.tv/.MainActivity
adb -s TV_SERIAL shell input keyevent KEYCODE_HOME
```

Keep the existing launcher installed and enabled. Roll back by running the
same `set-home-activity` command with the exact original component captured
above. Changing the default is a device setup step; installing an APK alone
does not change the user's default Home.
Verify the actual Home key afterward: some Google TV firmware accepts the role
change but still resolves Home to a privileged OEM activity with higher intent
priority. A successful shell response alone does not establish a working Home
replacement. Waking from standby can also resume whichever app was previously
foreground; changing Home does not override that TV behavior.

### Optional Home button helper

Version 0.1.2 includes an optional helper for Google TV firmware that ignores
the default-Home selection. It is **off by default**. Open **Apps → Home button
helper**, or the live overlay's **Settings → Home button helper**, to review its
purpose and open Android Accessibility settings. Enabling **Aviary Home button
helper** there is a separate, explicit owner decision. The app never enables
it automatically or writes global settings.

When enabled, it responds only to a window-state event whose package is
`com.google.android.apps.tv.launcherx` and activity is exactly
`com.google.android.apps.tv.launcherx.home.HomeActivity`. It opens Aviary using
an explicit Home intent. The Google TV screen may appear briefly first. Other
activities, other applications, Settings, clicks and key events do not trigger
it. Events older than two seconds and duplicate event deliveries are ignored.
A fresh Home press is accepted even immediately after the preceding one.
`notificationTimeout` is zero so framework coalescing cannot replace the exact
Home activity event with a generic window event; duplicate timestamps are
discarded by the helper itself.

The helper reads only event type, package/activity names and timing. It does not
read event text or request window contents, intercept keys, perform gestures,
record activity or send data. Its manifest requests no content, key-filtering
or gesture capability. It has no background playback, polling or delayed work.
Android's service permission protects its binding. Disabling it in Android
Accessibility settings restores ordinary Google TV Home behavior without
uninstalling anything. Google TV and Setup Wizard stay enabled.

The helper passed ten JVM tests, lint, and three instrumented
checks on the TCL using fabricated events without granting Accessibility.
Those checks cover the exact event filter, lifecycle, launch denial handling,
Home intent and manifest capability limits. Accessibility settings were
unchanged before and after those fabricated-event tests. They did not enable
the service.

After explicit owner approval, the helper was enabled through the TCL's normal
Accessibility confirmation on 2026-10-29. Physical Home → Aviary, Settings →
Home and VLC → Home passed. Settings and VLC each remained foreground for an
eight-second observation without the helper opening Aviary; root Back remained
in Aviary for eight seconds. Version 0.1.2 also passed three rapid Home presses,
settling in Aviary within 2.63 seconds of the batch beginning and remaining
there. A prior broad cooldown could swallow the final Home press; the fix
rejects only duplicate/older events and preserves each fresh activity event.
The installed helper is enabled on this authorized TV. All previously enabled
Accessibility services were preserved, and Google TV and Setup Wizard remain
enabled. Cold-boot and standby behavior were not tested.

Developer instrumentation can stop the app process and leave Android marking
the helper enabled but unbound. After such a test, toggle only this helper off
and on in Android Accessibility settings and verify a real Home press. Do not
infer active service binding from the enabled-service list alone.

Fullscreen controls hide after six seconds. Press a D-pad direction or OK to
show them. The first press wakes the controls; the next selects an action.
Live playback always reconnects at the current live edge. A replay retains its
position and pause state when the app briefly goes into the background.

The default grid plays up to four live streams simultaneously. A decoder error
switches the affected tile to a clearly labelled **Preview / 5s**. Previews use
screened server snapshots and can show **Preview unavailable** while screening
is paused. **Grid: previews** intentionally uses one live decoder for the
selected camera and snapshots for the others. This setting persists across
launches. Fullscreen uses one decoder in either mode.

Discovery browses only `_aviary._tcp`; it does not scan a subnet or other
ports. A matching advertisement must pass the unauthenticated static identity
check `{"service":"aviary-observatory","api":1}`. Discovery stops on screen
changes and on its ten-second deadline. Older Android Wi-Fi implementations
receive a scoped multicast lock while discovery runs.

The TV and server must be able to communicate on the same LAN. Guest Wi-Fi
isolation can prevent that. The local server needs a stable address, ideally a
DHCP reservation. HTTPS server URLs are also accepted, but a tailnet-only name
will only work on a TV with an appropriate network route. For ordinary TV use,
enter the LAN address instead.

## Build and verify

Use an isolated build directory on the designated Linux build host. Java 21,
Android SDK platform 37 and build tools are required. The Gradle wrapper pins
Gradle 9.8.1 with its distribution SHA-256. AGP is 9.4.0; Media3 is 1.11.1.

```sh
cd observatory/android-tv
export ANDROID_HOME=/path/to/Android/Sdk
./gradlew --no-daemon :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
```

The installable APK is `app/build/outputs/apk/debug/app-debug.apk`.
The debug APK is for private sideloading, not a Play Store release. Keep the
debug signing key consistent on the build host for in-place updates. A release
signing identity and distribution channel have not been configured.

The default instrumented checks use a dedicated Android TV emulator and do not
require a physical TV or the private deployment. They serve synthetic H.264/AAC
fixtures from an in-process HTTP server and exercise login, session cookies,
transient HLS failure, video decoding, remote focus, single-source audio,
fullscreen Back, MP4 seeking, pause/resume, lifecycle cleanup, preview mode and
expired-session recovery.

```sh
ANDROID_SERIAL=emulator-PORT ./gradlew --no-daemon :app:connectedDebugAndroidTest
```

`realServerSmokeWhenConfigured` is skipped unless `smoke-connection.json`
exists in the debug app's private files directory. For an authorized test only,
provide `{"server":"<local server URL>","pin":"<home passcode>"}` through a
private local file, never source control or command-line arguments. Remove the
file after testing. The smoke check logs in, decodes live HLS, opens fullscreen
and plays a recording over the real authenticated API.

`physicalDiscoveryAndSmokeWhenConfigured` is an additional explicitly selected
device test using the same private file. It exercises real NSD discovery, masked
passcode entry, simultaneous decoding, offline Home access, configurable app
pins, native Settings launch, player release and return to Aviary. Run it only
on the owner's authorized test TV; it opens Settings and changes then restores
Aviary's pin preferences. It does not change the default Home.

Two focused opt-in checks use an already saved, authorized deployment session:
`aviaryPhysicalHomeMute=true` enables `homeMuteOnEveryVisitUsingSavedConnection`,
which checks zero Home player volumes, explicit single-camera unmute, unchanged
players on an already-visible Home intent, and unchanged TV/system volume.
`aviaryPhysicalArrange=true` enables `physicalAppArrangeKeepsPlayersAndSound`,
which injects long-OK/D-pad/Back, checks persistent order with the same players,
selected camera and sound during moves, and restores the current pin preferences.
Both passed on the authorized TCL; they remain skipped unless explicitly enabled.

`aviaryPhysicalGeometry=true` enables `physicalWallGeometryAndWholeFrame` with
the same saved session. It checks that the three tiles cover the wall with no
gaps, video fills each tile without aspect distortion, full screen fits the whole
frame, D-pad follows the layout, and arrange mode retains the same players. It
captures normal, full-screen and arrange views without changing the owner's pins.
Version 0.1.5 passed this check on the authorized TCL: a 1920×872 wall with
1280×872 / 640×436 / 640×436 tiles, a 208-pixel app strip, and three decoded
streams. The test also verified camera-to-app navigation after a strip-only
refresh. The approved Home helper was rebound afterward and a real Home press
returned to Aviary; Home remained muted.

The synthetic test assets can be regenerated with FFmpeg:

```sh
ffmpeg -f lavfi -i testsrc2=size=320x180:rate=10 \
  -f lavfi -i sine=frequency=440:sample_rate=44100 -t 20 \
  -c:v libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p -g 20 \
  -c:a aac -b:a 32k -movflags +faststart test.mp4
ffmpeg -i test.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts test.ts
```

## Authentication and playback

The passcode is sent only to `POST /api/login` and is not saved. The returned
HttpOnly session is stored in app-private preferences; Android backup is
disabled. API and Media3 requests carry the same session cookie. Server and
media URLs must share an origin. HTTP onboarding accepts private IPv4,
loopback and `.local` hosts; remote URLs must use HTTPS. **Sign out** forgets
the TV's local session; it does not revoke sessions on other devices.

The app uses `/api/overview`, `/api/cameras/{id}/hls/live.m3u8`, screened
`/api/cameras/{id}/snapshot`, and `/api/clips` with `highlights`, `recordings`
and the server's pagination cursor. Clip video uses the API's authenticated
Range support. No backend changes or public ingress are required.

Network errors retry with a capped 1-15 second backoff. A 30-second buffering
watchdog retries stalled playback. A 401/403 returns to passcode entry.
Players and delayed work are released when the activity stops; live streams
are rebuilt on return. Screen-on is limited to the foreground activity.

## Validation limits

On 2026-10-29 the original TV APK, six JVM tests and Android lint completed on the Linux
build host. An Android TV API 30 emulator passed two fixture-backed flow tests;
the optional deployment test also passed against the real LAN API.

The authorized TCL Smart TV Pro running Android 14 passed the physical test:
automatic discovery, masked passcode, three simultaneous live decoders, one
selected audio source, D-pad fullscreen/Back, recording seek and pause/resume,
app pin order/hide persistence, offline Home app access, Settings launch with
player release, and playback on return. Private screenshots and full logs are
retained in ignored audit output. Three simultaneous decoders are established
for this tested device and stream format, not every Android TV.

Effective default-Home replacement did **not** pass on this TCL firmware. Both
the package command and native Home chooser saved Aviary as the selected role,
but Home still opened Google TV. A reversible authorized package-disable trial
then resolved Home to the OEM Setup Wizard recovery activity. That trial was
immediately rolled back without opening the recovery activity; Setup Wizard
was never changed. Android rejected the narrower request to disable only that
recovery component with a `SecurityException`. The original Google TV launcher and Home preference are
restored. The optional Home button helper above subsequently passed physical
handoff tests with owner approval. The system's default Home role remains
Google TV; the enabled helper opens Aviary when that Home activity appears.

Long-duration source continuity, a physical cold boot and acoustic output
measurement are separate acceptance checks. The test verifies only one player
has nonzero volume; it does not measure speaker output. Upstream recorder
restarts can still interrupt live playback. Preview mode remains available for
other devices with fewer decoder resources.

The native view and focus approach follows the existing local Android TV
application's explicit focus-neighbor and overscan patterns. Playback follows
the official [Media3 guide](https://developer.android.com/media/media3/exoplayer/hello-world)
and [TV navigation guidance](https://developer.android.com/training/tv/get-started/navigation).
