package org.aviary.tv

import android.content.Context
import android.os.SystemClock
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.media3.exoplayer.ExoPlayer
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

@RunWith(AndroidJUnit4::class)
class TvFlowTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext

    private fun views(view: View): List<View> = listOf(view) + if (view is ViewGroup) (0 until view.childCount).flatMap { views(view.getChildAt(it)) } else emptyList()
    private fun all(activity: MainActivity) = views(activity.findViewById(android.R.id.content))
    private fun await(label: String, timeout: Long = 20000, test: () -> Boolean) {
        val end = SystemClock.elapsedRealtime() + timeout
        while (SystemClock.elapsedRealtime() < end) { if (test()) return; SystemClock.sleep(150) }
        fail("Timed out: $label")
    }
    private fun field(activity: MainActivity, name: String): Any? = MainActivity::class.java.getDeclaredField(name).apply { isAccessible = true }.get(activity)
    private fun players(activity: MainActivity): List<ExoPlayer> = (field(activity, "slots") as List<*>).mapNotNull { slot ->
        slot!!.javaClass.getDeclaredField("player").apply { isAccessible = true }.get(slot) as ExoPlayer?
    }
    private fun click(scenario: ActivityScenario<MainActivity>, label: String) {
        scenario.onActivity { activity -> (all(activity).first { it is Button && it.text.toString() == label } as Button).performClick() }
    }
    private fun ready(scenario: ActivityScenario<MainActivity>, count: Int = 1) {
        await("$count decoded video players") {
            var okay = false
            scenario.onActivity { activity -> okay = players(activity).count { it.videoSize.width > 0 && it.playbackState == 3 } >= count }
            okay
        }
    }

    @Test fun onboardingDpadPlaybackReplayAndLifecycle() {
        FixtureServer().use { server ->
            context.getSharedPreferences("connection", Context.MODE_PRIVATE).edit().clear().commit()
            context.getSharedPreferences("MainActivity", Context.MODE_PRIVATE).edit().clear().commit()
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                scenario.onActivity { activity ->
                    (all(activity).first { it.tag == "server" } as EditText).setText(server.url)
                    (all(activity).first { it.tag == "passcode" } as EditText).setText("123456")
                }
                click(scenario, "Connect")
                ready(scenario, 2)
                assertTrue("HLS transient failure was retried", server.manifests.get() >= 3)
                scenario.onActivity { activity ->
                    assertEquals("camera-0", activity.currentFocus?.tag)
                    assertTrue(players(activity).all { it.volume == 0f })
                    val bottomCard = all(activity).first { it.tag == "camera-2" }
                    val position = IntArray(2); bottomCard.getLocationOnScreen(position)
                    assertTrue("All three camera tiles fit without scrolling", position[1] + bottomCard.height <= activity.resources.displayMetrics.heightPixels)
                }
                click(scenario, "Sound off")
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_RIGHT)
                scenario.onActivity { activity ->
                    assertEquals("camera-1", activity.currentFocus?.tag)
                    assertEquals(1, players(activity).count { it.volume > 0f })
                }
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
                ready(scenario)
                scenario.onActivity { activity -> assertEquals("full", field(activity, "mode")); assertEquals(1, players(activity).size) }
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
                ready(scenario, 2)
                scenario.onActivity { activity -> assertEquals("camera-1", activity.currentFocus?.tag) }
                click(scenario, "Replays")
                await("replay list") {
                    var found = false; scenario.onActivity { found = all(it).any { v -> v.tag == "clip-0" } }; found
                }
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
                ready(scenario)
                click(scenario, "Forward 10s")
                scenario.onActivity { activity -> assertTrue(players(activity).single().currentPosition >= 9000) }
                click(scenario, "Pause")
                scenario.onActivity { activity -> assertFalse(players(activity).single().playWhenReady) }
                scenario.moveToState(androidx.lifecycle.Lifecycle.State.CREATED)
                scenario.onActivity { activity -> assertTrue(players(activity).isEmpty()) }
                scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED)
                ready(scenario)
                scenario.onActivity { activity ->
                    assertFalse(players(activity).single().playWhenReady)
                    assertTrue(players(activity).single().currentPosition >= 9000)
                }
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
                scenario.onActivity { activity -> assertEquals("clips", field(activity, "mode")) }
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
                ready(scenario, 2)
                click(scenario, "Grid: live")
                ready(scenario)
                scenario.onActivity { activity -> assertEquals(1, players(activity).size); assertTrue(all(activity).any { it is TextView && it.text.contains("Preview") }) }
            }
            assertEquals("Only cookie auth accesses media", 0, server.unauthorized.get())
        }
    }

    @Test fun homeMuteOnEveryVisitUsingSavedConnection() {
        org.junit.Assume.assumeTrue("Physical Home mute check not requested",
            InstrumentationRegistry.getArguments().getString("aviaryPhysicalHomeMute") == "true")
        org.junit.Assume.assumeTrue("No saved deployment session", AviaryApi(context).connected)
        val cameraCount = AviaryApi(context).cameras().size.coerceAtMost(4)
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as android.media.AudioManager
        val originalSystemVolume = audioManager.getStreamVolume(android.media.AudioManager.STREAM_MUSIC)
        val observations = org.json.JSONObject()
        fun homeIntent() = android.content.Intent(context, MainActivity::class.java)
            .setAction(android.content.Intent.ACTION_MAIN).addCategory(android.content.Intent.CATEGORY_HOME)
            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_SINGLE_TOP)
        ActivityScenario.launch<MainActivity>(homeIntent()).use { scenario ->
            fun assertMuted(stage: String) {
                await("$stage muted Home players") {
                    var muted = false
                    scenario.onActivity { activity ->
                        muted = players(activity).size == cameraCount && players(activity).all { it.volume == 0f }
                    }
                    muted
                }
                scenario.onActivity { activity ->
                    val audio = all(activity).filterIsInstance<Button>().single { it.text == "Sound off" }
                    assertEquals("Sound off", audio.contentDescription)
                    observations.put(stage, org.json.JSONArray(players(activity).map { it.volume }))
                }
            }
            fun unmute() {
                click(scenario, "Sound off")
                scenario.onActivity { activity ->
                    assertEquals(1, players(activity).count { it.volume > 0f })
                    assertEquals("Sound on", all(activity).filterIsInstance<Button>().single { it.text == "Sound on" }.contentDescription)
                }
            }
            assertMuted("initial_home")
            unmute()
            var originalPlayers = emptyList<ExoPlayer>()
            scenario.onActivity { activity ->
                originalPlayers = players(activity)
                activity.startActivity(homeIntent())
            }
            assertMuted("home_already_visible")
            scenario.onActivity { activity -> assertEquals("Home mutes existing players without rebuilding", originalPlayers, players(activity)) }
            unmute()
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.CREATED)
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED)
            assertMuted("foreground_return")
            unmute()
            scenario.recreate()
            assertMuted("restored_activity")
            unmute()
            scenario.onActivity { it.startActivity(homeIntent()) }
            assertMuted("final_home")
            assertEquals(originalSystemVolume, audioManager.getStreamVolume(android.media.AudioManager.STREAM_MUSIC))
            observations.put("system_volume_unchanged", true).put("explicit_unmute_audible_players", 1)
            java.io.File(context.filesDir, "home-muted-result.json").writeText(observations.toString())
        }
    }

    @Test fun expiredSessionReturnsToPasscode() {
        FixtureServer(expired = true).use { server ->
            context.getSharedPreferences("connection", Context.MODE_PRIVATE).edit().putString("server", server.url).putString("session", "aviary_session=expired").commit()
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                await("expired-session onboarding") {
                    var found = false; scenario.onActivity { found = all(it).any { v -> v.tag == "passcode" } }; found
                }
                assertFalse(AviaryApi(context).connected)
            }
        }
    }

    @Test fun physicalDiscoveryAndSmokeWhenConfigured() {
        val config = java.io.File(context.filesDir, "smoke-connection.json")
        org.junit.Assume.assumeTrue("No private deployment supplied", config.exists())
        val data = org.json.JSONObject(config.readText())
        verifyOfflineHome()
        context.getSharedPreferences("connection", Context.MODE_PRIVATE).edit().clear().commit()
        context.getSharedPreferences("MainActivity", Context.MODE_PRIVATE).edit().clear().commit()
        fun capture(name: String) {
            instrumentation.uiAutomation.takeScreenshot().let { bitmap ->
                java.io.File(context.filesDir, "$name.png").outputStream().use { bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
                bitmap.recycle()
            }
        }
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        try {
            await("verified automatic server discovery", 18000) {
                var found = false
                scenario.onActivity { found = all(it).any { view -> view is TextView && view.text == "Aviary found. Enter your home passcode." } }
                found
            }
            scenario.onActivity { activity ->
                val server = all(activity).first { it.tag == "server" } as EditText
                assertEquals(View.GONE, server.visibility)
                assertTrue("Discovered the configured server", server.text.toString() == data.getString("server"))
                val pin = all(activity).first { it.tag == "passcode" } as EditText
                assertTrue("Passcode is masked on this TV", pin.transformationMethod is android.text.method.PasswordTransformationMethod)
                pin.setText(data.getString("pin"))
            }
            click(scenario, "Connect")
            await("physical sign-in") { AviaryApi(context).connected }
            val cameraCount = AviaryApi(context).cameras().size.coerceAtMost(4)
            ready(scenario, cameraCount)
            capture("physical-grid")
            val launcherPrefs = context.getSharedPreferences("launcher", Context.MODE_PRIVATE)
            val originalPins = launcherPrefs.getString("pins", "[]")!!
            val originalArray = org.json.JSONArray(originalPins)
            val pinIds = (0 until originalArray.length()).map { originalArray.getString(it) }
            assertTrue("A physical app row is available to configure", pinIds.size >= 2)
            val changedPins = AppPins.move(pinIds, pinIds[1], -1).dropLast(1)
            launcherPrefs.edit().putString("pins", org.json.JSONArray(changedPins).toString()).commit()
            scenario.recreate()
            ready(scenario)
            scenario.onActivity { activity ->
                val shown = (field(activity, "appButtons") as List<*>).filterIsInstance<View>()
                    .mapNotNull { (it.tag as? String)?.takeIf { tag -> tag.startsWith("app-") }?.removePrefix("app-") }
                    .filter { it != "com.android.tv.settings" }
                assertEquals(changedPins, shown)
            }
            launcherPrefs.edit().putString("pins", originalPins).commit()
            scenario.recreate()
            ready(scenario)
            click(scenario, "Apps")
            await("picker is reachable with the remote", 5000) {
                instrumentation.uiAutomation.rootInActiveWindow
                    ?.findAccessibilityNodeInfosByText("Choose apps for the row")?.isNotEmpty() == true
            }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            instrumentation.waitForIdleSync()
            scenario.onActivity { activity -> all(activity).first { it.tag == "camera-0" }.requestFocus() }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_UP)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_RIGHT)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_DOWN)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_RIGHT)
            scenario.onActivity { activity ->
                assertEquals("camera-1", activity.currentFocus?.tag)
                assertEquals(1, players(activity).count { it.volume > 0f })
            }
            SystemClock.sleep(1500)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
            ready(scenario)
            capture("physical-fullscreen-controls")
            SystemClock.sleep(6500)
            capture("physical-fullscreen")
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
            scenario.onActivity { activity -> assertEquals("full", field(activity, "mode")) }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            ready(scenario, cameraCount)
            scenario.onActivity { activity ->
                assertTrue(players(activity).all { it.volume == 0f })
                assertTrue(all(activity).any { it is Button && it.text == "Sound off" })
            }
            click(scenario, "Replays")
            click(scenario, "Recordings")
            await("physical recording list") {
                var found = false; scenario.onActivity { found = all(it).any { view -> view.tag == "clip-0" } }; found
            }
            capture("physical-recordings")
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
            ready(scenario)
            click(scenario, "Forward 10s")
            click(scenario, "Pause")
            scenario.onActivity { activity ->
                assertTrue(players(activity).single().currentPosition >= 9000)
                assertFalse(players(activity).single().playWhenReady)
            }
            capture("physical-replay")
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.CREATED)
            scenario.onActivity { activity -> assertTrue(players(activity).isEmpty()) }
            scenario.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED)
            ready(scenario)
            scenario.onActivity { activity ->
                assertTrue(players(activity).single().currentPosition >= 9000)
                assertFalse(players(activity).single().playWhenReady)
            }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            ready(scenario, cameraCount)
            scenario.onActivity { activity ->
                val settings = all(activity).first { it.tag == "app-com.android.tv.settings" }
                settings.requestFocus(); settings.performClick()
            }
            await("players released while another app is foreground") {
                var stopped = false; scenario.onActivity { stopped = players(it).isEmpty() }; stopped
            }
            android.os.ParcelFileDescriptor.AutoCloseInputStream(instrumentation.uiAutomation.executeShellCommand(
                "am start -n org.aviary.tv/.MainActivity")).use { it.readBytes() }
            ready(scenario)
            scenario.onActivity { activity -> assertEquals(cameraCount, players(activity).size) }
            capture("physical-home-return")
            java.io.File(context.filesDir, "physical-result.json").writeText(org.json.JSONObject()
                .put("model", android.os.Build.MODEL).put("android", android.os.Build.VERSION.RELEASE)
                .put("auto_discovery", true).put("live_decoders", cameraCount).put("audible_players", 1)
                .put("fullscreen_back", true).put("replay_seek", true).put("lifecycle_resume", true)
                .put("app_strip_settings_launch", true).put("launcher_return", true)
                .put("app_hide_reorder_persist", true).put("offline_home_apps", true).toString())
            scenario.onActivity { it.finish() }
            instrumentation.waitForIdleSync()
            SystemClock.sleep(300)
        } finally {
            // TCL's external launcher task transition invalidates ActivityScenario's
            // close() bookkeeping. Finish actual target activities through the lifecycle
            // registry instead; feature assertions above still propagate normally.
            instrumentation.runOnMainSync {
                val monitor = androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry.getInstance()
                for (stage in androidx.test.runner.lifecycle.Stage.values()) {
                    if (stage != androidx.test.runner.lifecycle.Stage.DESTROYED) {
                        monitor.getActivitiesInStage(stage).filterIsInstance<MainActivity>().forEach { it.finish() }
                    }
                }
            }
            instrumentation.waitForIdleSync()
        }
    }

    private fun verifyOfflineHome() {
        context.getSharedPreferences("connection", Context.MODE_PRIVATE).edit()
            .putString("server", "http://127.0.0.1:9").putString("session", "aviary_session=offline-fixture").commit()
        fun homeIntent() = android.content.Intent(context, MainActivity::class.java)
            .setAction(android.content.Intent.ACTION_MAIN).addCategory(android.content.Intent.CATEGORY_HOME)
            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        ActivityScenario.launch<MainActivity>(homeIntent()).use { scenario ->
            await("app strip while server is unreachable") {
                var found = false; scenario.onActivity { found = all(it).any { view -> view.tag == "app-com.android.tv.settings" } }; found
            }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            scenario.onActivity { assertFalse("Home Back must not finish into a relaunch loop", it.isFinishing) }
            scenario.onActivity { activity -> all(activity).first { it.tag == "app-com.android.tv.settings" }.performClick() }
            context.startActivity(homeIntent())
            await("Home remains usable on offline return") {
                var found = false; scenario.onActivity { found = all(it).any { view -> view.tag == "app-com.android.tv.settings" } }; found
            }
        }
        context.getSharedPreferences("connection", Context.MODE_PRIVATE).edit().clear().commit()
        ActivityScenario.launch<MainActivity>(homeIntent()).use { scenario ->
            await("app strip before authentication") {
                var found = false; scenario.onActivity { found = all(it).any { view -> view.tag == "app-com.android.tv.settings" } }; found
            }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            scenario.onActivity { assertFalse("Unauthenticated Home must remain available", it.isFinishing) }
        }
    }

    /** Optional, enabled by an ignored local file placed into the debug app's private directory. */
    @Test fun realServerSmokeWhenConfigured() {
        val config = java.io.File(context.filesDir, "smoke-connection.json")
        org.junit.Assume.assumeTrue("No private deployment supplied", config.exists())
        val data = org.json.JSONObject(config.readText())
        context.getSharedPreferences("MainActivity", Context.MODE_PRIVATE).edit().clear().commit()
        val api = AviaryApi(context)
        api.login(data.getString("server"), data.getString("pin"))
        assertTrue(api.cameras().isNotEmpty())
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            ready(scenario)
            scenario.onActivity { activity -> all(activity).first { it.tag == "camera-0" }.performClick() }
            ready(scenario)
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            click(scenario, "Replays")
            click(scenario, "Recordings")
            await("real recording list") {
                var found = false; scenario.onActivity { found = all(it).any { v -> v.tag == "clip-0" } }; found
            }
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_DPAD_CENTER)
            ready(scenario)
        }
    }

    private inner class FixtureServer(val expired: Boolean = false) : AutoCloseable {
        private val socket = ServerSocket(0)
        private val workers = Executors.newCachedThreadPool()
        val url = "http://127.0.0.1:${socket.localPort}"
        val manifests = AtomicInteger()
        val unauthorized = AtomicInteger()
        private val video = instrumentation.context.assets.open("test.mp4").use { it.readBytes() }
        private val segment = instrumentation.context.assets.open("test.ts").use { it.readBytes() }
        init { workers.execute { while (!socket.isClosed) try { val client = socket.accept(); workers.execute { respond(client) } } catch (_: Exception) { } } }
        private fun respond(client: Socket) = client.use {
            try {
                val reader = client.getInputStream().bufferedReader()
                val first = reader.readLine().split(' ')
                val path = first[1]
                val headers = mutableMapOf<String, String>()
                while (true) { val line = reader.readLine() ?: break; if (line.isEmpty()) break; headers[line.substringBefore(':').lowercase()] = line.substringAfter(':').trim() }
                var code = 200
                var type = "application/json"
                var extra = ""
                var content = "{}".toByteArray()
                if (path == "/api/login") extra = "Set-Cookie: aviary_session=test-session; HttpOnly; Path=/\r\n"
                else if (expired || headers["cookie"] != "aviary_session=test-session") { code = 401; unauthorized.incrementAndGet() }
                else when {
                    path == "/api/overview" -> content = """{"cameras":[{"id":"one","name":"Garden","online":true},{"id":"two","name":"Perch","online":true},{"id":"three","name":"Canopy","online":true}]}""".toByteArray()
                    path.startsWith("/api/clips?") -> content = """{"clips":[{"id":"sample","time":"2026-01-01T03:00:00Z","camera":"one","duration":20,"video":"/api/clips/sample/video","caption":"Test recording"}],"next":null}""".toByteArray()
                    path.endsWith("live.m3u8") -> {
                        if (manifests.incrementAndGet() == 1) code = 503
                        else { type = "application/vnd.apple.mpegurl"; content = ("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:20\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:20.0,\nsegment.ts\n").toByteArray() }
                    }
                    path.endsWith("segment.ts") -> { type = "video/mp2t"; content = segment }
                    path.endsWith("/video") -> {
                        type = "video/mp4"; content = video
                        val range = headers["range"]?.substringAfter("bytes=")?.substringBefore('-')?.toIntOrNull()
                        if (range != null && range < video.size) { code = 206; content = video.copyOfRange(range, video.size); extra = "Content-Range: bytes $range-${video.size - 1}/${video.size}\r\n" }
                    }
                    path.endsWith("/snapshot") -> code = 503
                    else -> code = 404
                }
                val head = "HTTP/1.1 $code OK\r\nContent-Type: $type\r\nContent-Length: ${content.size}\r\nConnection: close\r\n$extra\r\n"
                client.getOutputStream().write(head.toByteArray()); client.getOutputStream().write(content); client.getOutputStream().flush()
            } catch (_: Exception) { }
        }
        override fun close() { socket.close(); workers.shutdownNow() }
    }
}
