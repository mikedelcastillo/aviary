# Camera zone references

Open **Feeder and water zones** from the live camera wall. Choose a camera, wait for the frozen screened picture, draw around a bowl or perch, name it, then save within five minutes. **Refresh picture** discards the current drawing. Changing cameras also clears it. The editor uses the picture's actual aspect ratio, including on a phone.

Saving retains exactly the JPEG shown in the editor. The rolling camera picture can update while someone draws without changing the saved reference. The worker compares that reference against later views before counting visits. A visit means a bird was near the marked area; it does not establish eating, drinking, health, or correct identity.

## Frozen reference contract

- Authenticated, same-origin `POST /api/cameras/{camera}/zone-reference` returns the screened JPEG, an opaque `X-Aviary-Zone-Reference` token, and its lifetime in `X-Aviary-Zone-Ttl`. Responses are private and `no-store`.
- `POST /api/zones` requires `reference_token` alongside `camera`, `name`, `kind`, and normalized `[x, y, width, height]`. The token belongs to the authenticated session and camera. It can save one zone once. Missing, expired, already-used, or mismatched tokens return an explicit refresh error. Legacy clients cannot silently save a different rolling image.
- Immutable drafts are valid for five minutes. The API reserves memory capacity before reading images: at most 16 drafts globally and four per session, with a two-MiB JPEG limit and a 16-million-pixel dimension limit. Expired entries are removed on subsequent draft operations. Refresh retires that session's preceding completed draft for the same camera. Restarting the API invalidates unfinished drafts.
- Missing, malformed, stale, or person-paused screening fails closed both when obtaining a picture and when saving. Screening metadata and JPEG must be fresh; a partially published pair is rejected. No raw frame endpoint is used by this editor. The existing person detector remains the screening mechanism; a clear result is a model decision.
- The complete reference file is written before inserting the enabled zone row. A failed write does not create an active zone; a failed database operation removes the new reference file. Saving does not reread the rolling JPEG or replace the frozen camera position.

## Camera position

Known fixed cameras (`ptz_supported: false`) do not make ONVIF position calls, even if an ONVIF URL is configured. For cameras with pan/tilt support, freezing uses read-only position requests. It waits for a newly captured screened frame between two stationary pose readings and requires their coordinates to agree within 0.002 normalized units. The frame wait is bounded to six seconds and the whole operation to 14 seconds. Manual movements through this API share a lock during capture; Stop remains available. The editor never commands a camera to move.

These checks do not establish that no external controller moved a camera between observations. Continued reference-image and pose checks in the worker still determine whether visits may be counted. Moving a camera or changing the scene can pause counting until the saved view is recovered or a new zone is marked. Existing saved zones are left intact; this change does not repair old references retrospectively.

## Verification

`observatory/tests/test_zone_reference.py` uses fictional images and mocked camera calls. It checks exact-byte preservation after rolling frames change, delayed pose responses, changed poses, person/stale/missing states, session and camera binding, expiry, single use, memory quotas, and failed writes. `test_home.py` covers the authenticated API and legacy-request rejection.

`observatory/tests/browser_zones.mjs` exercises the actual built app at a local static test URL with every API request intercepted. It checks mobile touch drawing, matching the real image aspect ratio, the submitted reference token, camera changes, refresh, expiry, and screening failure. It never writes a production zone or contacts a camera. Example on an isolated test machine with Playwright installed:

```sh
AVIARY_URL=http://127.0.0.1:8766 node browser_zones.mjs
```

Keep generated screenshots, local runtime configuration, and captured images out of source control.
