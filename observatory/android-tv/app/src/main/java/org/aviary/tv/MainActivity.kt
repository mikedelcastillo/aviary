package org.aviary.tv

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.ComponentName
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.datasource.HttpDataSource
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.PlayerView
import java.util.concurrent.Executors
import kotlin.math.roundToInt

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
class MainActivity : Activity() {
    private lateinit var api: AviaryApi
    private val handler = Handler(Looper.getMainLooper())
    private val io = Executors.newFixedThreadPool(4)
    private val slots = mutableListOf<Slot>()
    private val cards = mutableListOf<View>()
    private var generation = 0
    private var started = false
    private var mode = "connect"
    private var cameras = emptyList<Camera>()
    private var selected = 0
    private var sound = false
    private var audioGranted = false
    private val audioManager by lazy { getSystemService(AUDIO_SERVICE) as AudioManager }
    private val audioRequest by lazy {
        AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(android.media.AudioAttributes.Builder().setUsage(android.media.AudioAttributes.USAGE_MEDIA).build())
            .setOnAudioFocusChangeListener({ change ->
                audioGranted = change == AudioManager.AUDIOFOCUS_GAIN
                updateVolumes()
            }, handler).build()
    }
    private var allLive = true
    private var clipView = "highlights"
    private var clipPage = ClipPage(emptyList(), null)
    private val cursors = mutableListOf<String?>(null)
    private var clipIndex = 0
    private var fullClip: Clip? = null
    private var replayPosition = 0L
    private var replayPlaying = true
    private var fullStatus: TextView? = null
    private var fullOverlay: View? = null
    private var fullBack: View? = null
    private var playButton: Button? = null
    private val hideControls = Runnable { fullOverlay?.visibility = View.GONE }
    private var status: TextView? = null
    private var discovery: AviaryDiscovery? = null
    private var wallOverlay: View? = null
    private var wallFirst: View? = null
    private val appButtons = mutableListOf<View>()
    private val hideWallControls = Runnable {
        wallOverlay?.visibility = View.GONE
        cards.getOrNull(selected)?.requestFocus()
    }
    private var root: LinearLayout? = null
    private val green = Color.rgb(181, 215, 171)
    private val backgroundColor = Color.rgb(16, 25, 19)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        api = AviaryApi(this)
        allLive = getPreferences(MODE_PRIVATE).getBoolean("allLive", true)
        mode = if (api.connected) "grid" else "connect"
        savedInstanceState?.let {
            selected = it.getInt("selected", 0)
            sound = it.getBoolean("sound", false)
        }
        if (Build.VERSION.SDK_INT >= 33) onBackInvokedDispatcher.registerOnBackInvokedCallback(0) { goBack() }
    }

    override fun onStart() {
        super.onStart()
        started = true
        when (mode) {
            "connect" -> showConnect()
            "full" -> showFull()
            "clips" -> showClips(false)
            else -> if (cameras.isEmpty()) loadCameras() else showGrid()
        }
    }

    override fun onNewIntent(newIntent: Intent) {
        super.onNewIntent(newIntent)
        intent = newIntent
        if (newIntent.hasCategory(Intent.CATEGORY_HOME) && api.connected) {
            val alreadyHome = mode == "grid"
            mode = "grid"
            if (started && (!alreadyHome || slots.isEmpty())) {
                if (cameras.isEmpty()) loadCameras() else showGrid()
            }
        }
    }

    override fun onStop() {
        if (mode == "full" && fullClip != null) slots.firstOrNull()?.player?.let {
            replayPosition = it.currentPosition
            replayPlaying = it.playWhenReady
        }
        started = false
        audioManager.abandonAudioFocusRequest(audioRequest); audioGranted = false
        clearScreen()
        super.onStop()
    }

    override fun onDestroy() { io.shutdownNow(); super.onDestroy() }
    override fun onSaveInstanceState(outState: Bundle) {
        outState.putInt("selected", selected)
        outState.putBoolean("sound", sound)
        super.onSaveInstanceState(outState)
    }
    @Deprecated("Older Android TV Back key")
    override fun onBackPressed() { goBack() }

    private fun goBack() {
        if (mode == "connect" && currentFocus is EditText) {
            (getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager)
                .hideSoftInputFromWindow(currentFocus?.windowToken, 0)
            appButtons.firstOrNull()?.requestFocus()
            return
        }
        when (mode) {
            "full" -> if (fullClip == null) showGrid() else showClips(false)
            "clips" -> showGrid()
            "connect" -> if (api.connected) loadCameras()
                         else if (intent.hasCategory(Intent.CATEGORY_HOME)) appButtons.firstOrNull()?.requestFocus()
                         else finish()
            else -> if (wallOverlay?.visibility == View.VISIBLE) hideWallControls.run()
                    else if (!intent.hasCategory(Intent.CATEGORY_HOME)) finish()
        }
    }

    private fun px(value: Int) = (value * resources.displayMetrics.density).roundToInt()
    private fun text(value: String, size: Float = 18f, bold: Boolean = false) = TextView(this).apply {
        text = value
        textSize = size
        setTextColor(Color.rgb(237, 241, 233))
        typeface = Typeface.create("sans-serif", if (bold) Typeface.BOLD else Typeface.NORMAL)
        includeFontPadding = false
    }
    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    private fun row() = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    private fun fill(color: Int, stroke: Int = color, width: Int = 1) = GradientDrawable().apply {
        setColor(color); cornerRadius = px(8).toFloat(); setStroke(px(width), stroke)
    }
    private fun focusBackground() = StateListDrawable().apply {
        addState(intArrayOf(android.R.attr.state_focused), fill(Color.rgb(48, 67, 47), green, 3))
        addState(intArrayOf(), fill(Color.rgb(28, 40, 30), Color.rgb(60, 77, 58)))
    }
    private fun button(label: String, action: () -> Unit) = Button(this).apply {
        id = View.generateViewId(); text = label; contentDescription = label
        isFocusableInTouchMode = true
        textSize = 16f; isAllCaps = false; setTextColor(Color.WHITE)
        minHeight = px(44); minimumHeight = px(44); minWidth = 0; minimumWidth = 0
        setPadding(px(18), px(8), px(18), px(8)); stateListAnimator = null
        background = focusBackground(); setOnClickListener { action() }
        layoutParams = LinearLayout.LayoutParams(-2, px(46)).apply { marginEnd = px(10) }
    }
    private fun clearScreen() {
        discovery?.stop(); discovery = null
        generation++
        handler.removeCallbacksAndMessages(null)
        slots.forEach { it.release() }; slots.clear(); cards.clear()
        fullStatus = null
        fullOverlay = null; fullBack = null
        playButton = null
        wallOverlay = null; wallFirst = null
        appButtons.clear()
    }
    private fun page(title: String): LinearLayout {
        clearScreen()
        val layout = column().apply {
            setBackgroundColor(backgroundColor)
            // A 5% safe area keeps controls inside older TV overscan.
            val inset = (resources.displayMetrics.widthPixels * .05f).roundToInt()
            val verticalInset = (resources.displayMetrics.heightPixels * .045f).roundToInt()
            setPadding(inset, verticalInset, inset, verticalInset)
        }
        val header = row()
        header.addView(text("AVIARY", 26f, true), LinearLayout.LayoutParams(0, -2, 1f))
        header.addView(text(title, 18f))
        layout.addView(header, LinearLayout.LayoutParams(-1, px(42)))
        val shell = column().apply { setBackgroundColor(backgroundColor) }
        shell.addView(layout, LinearLayout.LayoutParams(-1, 0, 1f))
        shell.addView(appStrip(), LinearLayout.LayoutParams(-1, px(94)))
        setContentView(shell); root = layout
        return layout
    }
    private fun note(parent: LinearLayout, value: String): TextView = text(value, 14f).apply {
        setTextColor(green); setPadding(0, px(8), 0, px(8))
        parent.addView(this, LinearLayout.LayoutParams(-1, -2))
    }
    private fun <T> async(work: () -> T, complete: (T) -> Unit, failure: (Exception) -> Unit = { showError(it) }) {
        val ticket = generation
        io.execute {
            try {
                val value = work()
                handler.post { if (started && ticket == generation) complete(value) }
            } catch (error: Exception) {
                handler.post { if (started && ticket == generation) failure(error) }
            }
        }
    }
    private fun showError(error: Exception) {
        if (error is LoginRequired) { api.signOut(); showConnect(error.message ?: "Sign in again.") }
        else status?.text = error.message?.take(150) ?: "Could not reach the server. Try again."
    }

    private fun showConnect(message: String = "Looking for Aviary on your home network...") {
        mode = "connect"
        val parent = page("Connect your TV")
        val form = column().apply { setPadding(px(60), px(12), px(60), 0) }
        val formScroll = ScrollView(this).apply { isFocusable = false; addView(form) }
        parent.addView(formScroll, LinearLayout.LayoutParams(-1, 0, 1f))
        form.addView(text("Your aviary, on the big screen", 28f, true))
        status = note(form, message)
        val discoveryControls = row()
        val rescan = button("Scan again") { showConnect() }
        discoveryControls.addView(rescan)
        form.addView(discoveryControls)
        val addressLabel = text("Local server URL", 15f).apply { visibility = View.GONE }
        form.addView(addressLabel)
        val server = EditText(this).apply {
            id = View.generateViewId(); tag = "server"
            setText(api.base); hint = "http://192.168.x.x:8740"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(); textSize = 19f; setTextColor(Color.WHITE)
            setSelectAllOnFocus(true)
            visibility = View.GONE
        }
        form.addView(server, LinearLayout.LayoutParams(-1, px(52)))
        form.addView(text("Home passcode", 15f))
        val pin = EditText(this).apply {
            id = View.generateViewId(); tag = "passcode"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            setSingleLine(); textSize = 19f; setTextColor(Color.WHITE)
            transformationMethod = android.text.method.PasswordTransformationMethod.getInstance()
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_DONE
        }
        form.addView(pin, LinearLayout.LayoutParams(-1, px(52)))
        var manualMode = false
        val manual = button("Enter address") {
            manualMode = true
            discovery?.stop(); discovery = null
            addressLabel.visibility = View.VISIBLE; server.visibility = View.VISIBLE
            pin.nextFocusUpId = server.id
            status?.text = "Enter your server address and home passcode."
            server.requestFocus()
        }
        discoveryControls.addView(manual)
        lateinit var connect: Button
        connect = button("Connect") {
            val address = server.text.toString()
            val code = pin.text.toString()
            if (code.isBlank()) { status?.text = "Enter your home passcode."; pin.requestFocus() }
            else {
                connect.isEnabled = false; status?.text = "Connecting..."
                async({ api.login(address, code) }, {
                    pin.text.clear()
                    (getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager).hideSoftInputFromWindow(pin.windowToken, 0)
                    loadCameras()
                }, { error -> connect.isEnabled = true; status?.text = error.message ?: "Could not connect."; connect.requestFocus() })
            }
        }
        form.addView(connect)
        pin.setOnEditorActionListener { _, action, _ ->
            if (action == android.view.inputmethod.EditorInfo.IME_ACTION_DONE) { connect.performClick(); true } else false
        }
        note(form, "Aviary finds its server on the same home network. No VPN app is needed.")
        server.nextFocusDownId = pin.id; pin.nextFocusUpId = server.id; pin.nextFocusDownId = connect.id
        connect.nextFocusUpId = pin.id
        pin.nextFocusUpId = rescan.id
        rescan.nextFocusDownId = pin.id; manual.nextFocusDownId = pin.id
        var matched = false
        val ticket = generation
        discovery = AviaryDiscovery(this, { candidate ->
            async({ api.verifyDiscovery(candidate) }, { address ->
                if (!matched && !manualMode && ticket == generation) {
                    matched = true; server.setText(address)
                    status?.text = "Aviary found. Enter your home passcode."
                    discovery?.stop(); discovery = null
                    pin.requestFocus()
                }
            }, { /* A service advertisement is only trusted after HTTP identity validation. */ })
        }, {
            if (!matched && ticket == generation) status?.text = if (api.base.isNotEmpty())
                "No new server found. Your saved connection is ready, or scan again."
                else "No Aviary server found. Check the home network, then scan again."
        }).also { it.start() }
        rescan.requestFocus()
    }

    private fun loadCameras() {
        mode = "grid"
        val parent = page("Cameras")
        status = note(parent, "Connecting to your aviary...")
        parent.addView(button("Retry") { loadCameras() })
        parent.addView(button("Connection settings") { showConnect() })
        async({ api.cameras() }, { cameras = it; showGrid() })
    }

    private fun showGrid() {
        mode = "grid"; fullClip = null
        clearScreen()
        val frame = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        setContentView(frame)
        val home = column()
        frame.addView(home, FrameLayout.LayoutParams(-1, -1))
        val grid = column()
        home.addView(grid, LinearLayout.LayoutParams(-1, 0, 1f))
        home.addView(appStrip(), LinearLayout.LayoutParams(-1, px(94)))
        val clock = text("", 20f, true).apply {
            setPadding(px(10), px(5), px(10), px(5)); setBackgroundColor(0xB0101913.toInt())
        }
        frame.addView(clock, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END).apply {
            topMargin = px(12); marginEnd = px(24)
        })
        fun updateClock() {
            if (mode != "grid" || !started) return
            clock.text = android.text.format.DateFormat.getTimeFormat(this).format(java.util.Date())
            handler.postDelayed({ updateClock() }, 1000)
        }
        updateClock()
        val overlay = column().apply {
            setPadding(px(32), px(16), px(32), px(12))
            setBackgroundColor(0xEB101913.toInt()); visibility = View.GONE
        }
        frame.addView(overlay, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
        wallOverlay = overlay
        val toolbar = row()
        val replay = button("Replays") { cursors.clear(); cursors.add(null); showClips(true) }
        lateinit var audio: Button
        audio = button(if (sound) "Sound on" else "Sound off") { sound = !sound; audio.text = if (sound) "Sound on" else "Sound off"; applyAudio() }
        val gridMode = button(if (allLive) "Grid: live" else "Grid: previews") {
            allLive = !allLive; getPreferences(MODE_PRIVATE).edit().putBoolean("allLive", allLive).apply(); showGrid()
        }
        val refresh = button("Refresh") { loadCameras() }
        val settings = button("Settings") {
            AlertDialog.Builder(this).setTitle("Aviary TV").setItems(arrayOf("Change connection", "Home button helper", "Sign out")) { _, which ->
                when (which) {
                    1 -> showHomeHandoff()
                    2 -> { api.signOut(); showConnect() }
                    else -> showConnect()
                }
            }.setNegativeButton("Cancel", null).show()
        }
        val controls = listOf(replay, audio, gridMode, refresh, settings)
        controls.forEach { toolbar.addView(it) }; overlay.addView(toolbar)
        wallFirst = replay
        status = note(overlay, "").apply { visibility = View.GONE }
        if (cameras.isEmpty()) {
            status?.visibility = View.VISIBLE; status?.text = "No cameras are configured on this server."
            overlay.visibility = View.VISIBLE; refresh.requestFocus(); return
        }
        selected = selected.coerceIn(cameras.indices)
        val columns = kotlin.math.ceil(kotlin.math.sqrt(cameras.size.toDouble())).toInt().coerceAtLeast(1)
        cameras.chunked(columns).forEachIndexed { rowIndex, group ->
            val line = row()
            grid.addView(line, LinearLayout.LayoutParams(-1, 0, 1f))
            group.forEachIndexed { colIndex, camera ->
                val index = rowIndex * columns + colIndex
                val card = FrameLayout(this).apply {
                    id = View.generateViewId(); tag = "camera-$index"
                    isFocusable = true; isFocusableInTouchMode = true; isClickable = true
                    background = StateListDrawable().apply {
                        addState(intArrayOf(android.R.attr.state_focused), fill(Color.BLACK, green, 1))
                        addState(intArrayOf(), fill(Color.BLACK))
                    }
                    setPadding(px(1), px(1), px(1), px(1))
                    contentDescription = "${camera.name}, open full screen"
                    setOnClickListener { selected = index; fullClip = null; showFull() }
                    setOnFocusChangeListener { _, focused -> if (focused) { selected = index; applyAudio(); if (!allLive) startSelectedPreview() } }
                    setOnKeyListener { _, key, event ->
                        if (key == KeyEvent.KEYCODE_DPAD_UP && rowIndex == 0) {
                            if (event.action == KeyEvent.ACTION_DOWN) showWallControls()
                            true
                        } else false
                    }
                }
                line.addView(card, LinearLayout.LayoutParams(0, -1, 1f))
                cards += card
                val image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(Color.BLACK) }
                card.addView(image, FrameLayout.LayoutParams(-1, -1))
                val playerView = playerView()
                card.addView(playerView, FrameLayout.LayoutParams(-1, -1))
                val state = text("Connecting", 12f).apply { setBackgroundColor(0xC0101913.toInt()); setPadding(px(8), px(4), px(8), px(4)) }
                card.addView(state, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END))
                slots += Slot(index, camera, playerView, state, image)
            }
            repeat(columns - group.size) { line.addView(View(this), LinearLayout.LayoutParams(0, -1, 1f)) }
        }
        cards.forEachIndexed { index, card ->
            card.nextFocusLeftId = if (index % columns > 0) cards[index - 1].id else card.id
            card.nextFocusRightId = if (index % columns < columns - 1) cards.getOrNull(index + 1)?.id ?: card.id else card.id
            card.nextFocusUpId = cards.getOrNull(index - columns)?.id ?: replay.id
            card.nextFocusDownId = cards.getOrNull(index + columns)?.id ?: appButtons.firstOrNull()?.id ?: card.id
        }
        controls.forEachIndexed { index, control ->
            control.nextFocusDownId = cards[selected].id
            control.nextFocusLeftId = controls.getOrNull(index - 1)?.id ?: control.id
            control.nextFocusRightId = controls.getOrNull(index + 1)?.id ?: control.id
        }
        appButtons.forEachIndexed { index, app ->
            app.nextFocusUpId = cards[selected].id
            app.nextFocusLeftId = appButtons.getOrNull(index - 1)?.id ?: app.id
            app.nextFocusRightId = appButtons.getOrNull(index + 1)?.id ?: app.id
            app.nextFocusDownId = app.id
        }
        slots.forEach { if (allLive && it.index < 4) it.startLive() else it.preview() }
        cards[selected].requestFocus()
        refreshSnapshots()
    }

    @Suppress("DEPRECATION")
    private fun populateApps(row: LinearLayout) {
        val candidates = (listOf(Intent.CATEGORY_LEANBACK_LAUNCHER, Intent.CATEGORY_LAUNCHER).flatMap { category ->
            packageManager.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(category), 0)
        } + listOfNotNull(packageManager.resolveActivity(Intent(android.provider.Settings.ACTION_SETTINGS), 0)))
            .distinctBy { it.activityInfo.packageName }.filter { it.activityInfo.packageName != packageName }
            .sortedBy { it.loadLabel(packageManager).toString().lowercase() }
        val configurable = candidates.filter { it.activityInfo.packageName != "com.android.tv.settings" }
        val available = configurable.map { it.activityInfo.packageName }
        val pins = AppPins.visible(readPins(), available)
        if (readPins() == null) writePins(pins)
        val manage = button("Apps") { showAppsMenu(candidates) }.apply { tag = "manage-apps" }
        row.addView(manage, LinearLayout.LayoutParams(px(112), -1).apply { marginEnd = px(8) })
        appButtons += manage
        val displayed = pins.mapNotNull { id -> configurable.firstOrNull { it.activityInfo.packageName == id } } +
            candidates.filter { it.activityInfo.packageName == "com.android.tv.settings" }
        for (info in displayed) {
            val label = info.loadLabel(packageManager).toString()
            val tile = column().apply {
                id = View.generateViewId(); tag = "app-${info.activityInfo.packageName}"
                gravity = Gravity.CENTER; isFocusable = true; isFocusableInTouchMode = true; isClickable = true
                background = focusBackground(); contentDescription = "Open $label"
                setPadding(px(8), px(5), px(8), px(5))
                setOnClickListener {
                    launchApp(info)
                }
                if (info.activityInfo.packageName != "com.android.tv.settings") setOnLongClickListener {
                    showPinActions(info.activityInfo.packageName, label); true
                }
            }
            tile.addView(ImageView(this).apply {
                setImageDrawable(info.loadIcon(packageManager)); scaleType = ImageView.ScaleType.FIT_CENTER
            }, LinearLayout.LayoutParams(px(42), px(42)))
            tile.addView(text(label, 12f).apply {
                gravity = Gravity.CENTER; maxLines = 1; ellipsize = android.text.TextUtils.TruncateAt.END
                setPadding(0, px(4), 0, 0)
            }, LinearLayout.LayoutParams(-1, -2))
            row.addView(tile, LinearLayout.LayoutParams(px(112), -1).apply { marginEnd = px(8) })
            appButtons += tile
        }
        appButtons.forEachIndexed { index, app ->
            app.nextFocusLeftId = appButtons.getOrNull(index - 1)?.id ?: app.id
            app.nextFocusRightId = appButtons.getOrNull(index + 1)?.id ?: app.id
            app.nextFocusDownId = app.id
        }
    }

    private fun readPins(): List<String>? = getSharedPreferences("launcher", MODE_PRIVATE).getString("pins", null)?.let { value ->
        runCatching { org.json.JSONArray(value).let { array -> (0 until array.length()).map { array.getString(it) } } }.getOrNull()
    }
    private fun writePins(pins: List<String>) {
        getSharedPreferences("launcher", MODE_PRIVATE).edit().putString("pins", org.json.JSONArray(pins).toString()).apply()
    }
    private fun refreshPage(focusTag: String? = null) {
        when (mode) {
            "connect" -> showConnect()
            "clips" -> showClips(false)
            else -> if (api.connected && cameras.isNotEmpty()) showGrid() else loadCameras()
        }
        if (focusTag != null) handler.post { appButtons.firstOrNull { it.tag == focusTag }?.requestFocus() }
    }
    private fun launchApp(info: android.content.pm.ResolveInfo) {
        val launch = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
            .setComponent(ComponentName(info.activityInfo.packageName, info.activityInfo.name))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
        runCatching { startActivity(launch) }.onFailure {
            Toast.makeText(this, "This app could not open.", Toast.LENGTH_SHORT).show()
        }
    }
    private fun showAppsMenu(apps: List<android.content.pm.ResolveInfo>) {
        val labels = arrayOf("Choose apps for the row", "Arrange the row", "Home button helper") + apps.map { it.loadLabel(packageManager).toString() }
        AlertDialog.Builder(this).setTitle("Apps").setItems(labels) { _, index ->
            when (index) {
                0 -> showAppPicker(apps.filter { it.activityInfo.packageName != "com.android.tv.settings" })
                1 -> {
                    val pinned = (readPins() ?: emptyList()).mapNotNull { id -> apps.firstOrNull { it.activityInfo.packageName == id } }
                    if (pinned.isEmpty()) showAppPicker(apps.filter { it.activityInfo.packageName != "com.android.tv.settings" })
                    else AlertDialog.Builder(this).setTitle("Arrange the row")
                        .setItems(pinned.map { it.loadLabel(packageManager).toString() }.toTypedArray()) { _, selected ->
                            val item = pinned[selected]; showPinActions(item.activityInfo.packageName, item.loadLabel(packageManager).toString())
                        }.setNegativeButton("Done", null).show()
                }
                2 -> showHomeHandoff()
                else -> launchApp(apps[index - 3])
            }
        }.setNegativeButton("Close", null).show()
    }
    private fun showHomeHandoff() {
        val manager = getSystemService(ACCESSIBILITY_SERVICE) as android.view.accessibility.AccessibilityManager
        val enabled = manager.getEnabledAccessibilityServiceList(android.accessibilityservice.AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
            .any { it.resolveInfo.serviceInfo.let { info -> info.packageName == packageName && info.name == HomeHandoffService::class.java.name } }
        AlertDialog.Builder(this).setTitle("Home button helper: ${if (enabled) "On" else "Off"}")
            .setMessage("Some TCL TVs keep Google TV as Home even after choosing Aviary. This optional helper opens Aviary when the Google TV Home screen appears.\n\n" +
                "It checks only that screen’s app and activity name. It does not read screen text, request window contents or intercept keys. It does not record activity or send data. Other apps and Settings are ignored.\n\n" +
                "Off by default. To use it, enable ‘Aviary Home button helper’ in Android Accessibility settings. Turn it off there at any time. Google TV and Setup Wizard stay installed and enabled.")
            .setPositiveButton("Open Accessibility settings") { _, _ ->
                runCatching { startActivity(Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS)) }.onFailure {
                    Toast.makeText(this, "Open Accessibility from the TV’s system Settings.", Toast.LENGTH_LONG).show()
                }
            }.setNegativeButton("Close", null).show()
    }
    private fun showAppPicker(apps: List<android.content.pm.ResolveInfo>) {
        val before = readPins() ?: emptyList()
        val selected = before.toMutableSet()
        val dialog = AlertDialog.Builder(this).setTitle("Choose apps for the row")
            .setMultiChoiceItems(apps.map { it.loadLabel(packageManager).toString() }.toTypedArray(),
                apps.map { it.activityInfo.packageName in selected }.toBooleanArray()) { _, index, checked ->
                val id = apps[index].activityInfo.packageName
                if (checked) selected.add(id) else selected.remove(id)
            }.setPositiveButton("Save") { _, _ ->
                writePins(AppPins.select(before, apps.map { it.activityInfo.packageName }.filter { it in selected }))
                refreshPage("manage-apps")
            }.setNegativeButton("Cancel", null).create()
        dialog.setOnShowListener {
            dialog.listView.setOnKeyListener { _, keyCode, event ->
                if (event.action == KeyEvent.ACTION_DOWN && keyCode == KeyEvent.KEYCODE_DPAD_RIGHT) {
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).requestFocus(); true
                } else if (event.action == KeyEvent.ACTION_DOWN && keyCode == KeyEvent.KEYCODE_DPAD_LEFT) {
                    dialog.getButton(AlertDialog.BUTTON_NEGATIVE).requestFocus(); true
                } else false
            }
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).nextFocusUpId = dialog.listView.id
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).nextFocusUpId = dialog.listView.id
        }
        dialog.show()
    }
    private fun showPinActions(id: String, label: String) {
        AlertDialog.Builder(this).setTitle(label).setItems(arrayOf("Move left", "Move right", "Hide from row")) { _, action ->
            val before = readPins() ?: emptyList()
            val updated = if (action == 2) before.filter { it != id } else AppPins.move(before, id, if (action == 0) -1 else 1)
            writePins(updated)
            val focus = if (action == 2) updated.getOrNull(before.indexOf(id).coerceAtMost(updated.lastIndex)) else id
            refreshPage(focus?.let { "app-$it" } ?: "manage-apps")
        }.setNegativeButton("Cancel", null).show()
    }

    private fun appStrip(): HorizontalScrollView {
        val apps = HorizontalScrollView(this).apply {
            isFocusable = false; isHorizontalScrollBarEnabled = false
            setBackgroundColor(backgroundColor)
        }
        val row = row().apply { setPadding(px(24), px(8), px(24), px(8)) }
        apps.addView(row)
        populateApps(row)
        return apps
    }

    private fun showWallControls() {
        wallOverlay?.visibility = View.VISIBLE
        wallFirst?.requestFocus()
        handler.removeCallbacks(hideWallControls); handler.postDelayed(hideWallControls, 6000)
    }

    private fun startSelectedPreview() {
        slots.forEach { if (it.index == selected) { if (it.player == null) it.startLive() } else if (it.player != null) it.preview() }
    }
    private fun applyAudio() {
        if (sound && !audioGranted) audioGranted = audioManager.requestAudioFocus(audioRequest) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        if (!sound && audioGranted) { audioManager.abandonAudioFocusRequest(audioRequest); audioGranted = false }
        updateVolumes()
    }
    private fun updateVolumes() { slots.forEach { it.player?.volume = if (sound && audioGranted && (mode == "full" || it.index == selected)) 1f else 0f } }

    private fun refreshSnapshots() {
        if (mode != "grid") return
        slots.filter { it.player == null && !it.snapshotBusy }.forEach { slot ->
            val camera = slot.camera ?: return@forEach
            slot.snapshotBusy = true
            async({ api.snapshot(camera) }, { bitmap ->
                slot.snapshotBusy = false
                if (slot.player == null) {
                    slot.image?.setImageBitmap(bitmap)
                    slot.state.text = if (bitmap == null) "Preview unavailable" else "Preview / 5s"
                }
            }, { error -> slot.snapshotBusy = false; if (error is LoginRequired) showError(error) else if (slot.player == null) slot.state.text = "Preview unavailable" })
        }
        handler.postDelayed({ refreshSnapshots() }, 5000)
    }

    private fun showClips(reload: Boolean) {
        mode = "clips"
        val parent = page("Replays")
        val toolbar = row()
        val live = button("Live cameras") { showGrid() }
        val highlights = button(if (clipView == "highlights") "Highlights selected" else "Highlights") {
            clipView = "highlights"; cursors.clear(); cursors.add(null); showClips(true)
        }
        val recordings = button(if (clipView == "recordings") "Recordings selected" else "Recordings") {
            clipView = "recordings"; cursors.clear(); cursors.add(null); showClips(true)
        }
        listOf(live, highlights, recordings).forEach { toolbar.addView(it) }; parent.addView(toolbar)
        status = note(parent, if (reload) "Loading replays..." else "Choose a recording. Back returns to live cameras.")
        if (reload) {
            parent.addView(button("Retry") { showClips(true) })
            async({ api.clips(clipView, cursors.last()) }, { clipPage = it; clipIndex = 0; showClips(false) })
            return
        }
        val scroll = ScrollView(this).apply { isFocusable = false }
        val list = column(); scroll.addView(list)
        parent.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        val entries = mutableListOf<View>()
        clipPage.clips.forEachIndexed { index, clip ->
            val cameraName = cameras.firstOrNull { it.id == clip.camera }?.name ?: clip.camera
            val title = "${formatTime(clip.time)}  /  $cameraName  /  ${clip.duration.roundToInt()}s\n${clip.caption}"
            val entry = button(title) { clipIndex = index; fullClip = clip; replayPosition = 0; replayPlaying = true; showFull() }.apply {
                tag = "clip-$index"; gravity = Gravity.CENTER_VERTICAL or Gravity.START; maxLines = 2
                ellipsize = android.text.TextUtils.TruncateAt.END
            }
            list.addView(entry, LinearLayout.LayoutParams(-1, px(72)).apply { bottomMargin = px(8) })
            entries += entry
        }
        if (entries.isEmpty()) status?.text = if (clipView == "highlights") "No approved highlights yet. Open Recordings to watch recent footage." else "No recordings in this page."
        val paging = row()
        if (cursors.size > 1) paging.addView(button("Newer") { cursors.removeAt(cursors.lastIndex); showClips(true) }.also { entries += it })
        if (clipPage.next != null) paging.addView(button("Older") { cursors.add(clipPage.next); showClips(true) }.also { entries += it })
        list.addView(paging)
        entries.forEachIndexed { index, entry ->
            entry.nextFocusUpId = entries.getOrNull(index - 1)?.id ?: highlights.id
            entry.nextFocusDownId = entries.getOrNull(index + 1)?.id ?: entry.id
        }
        listOf(live, highlights, recordings).forEach { it.nextFocusDownId = entries.firstOrNull()?.id ?: it.id }
        (entries.getOrNull(clipIndex) ?: recordings).requestFocus()
    }
    private fun formatTime(value: String): String = try {
        java.time.Instant.parse(value).atZone(java.time.ZoneId.systemDefault()).format(java.time.format.DateTimeFormatter.ofPattern("MMM d, h:mm a"))
    } catch (_: Exception) { value.take(19).replace('T', ' ') }

    private fun showFull() {
        mode = "full"
        clearScreen()
        val frame = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        setContentView(frame)
        val video = playerView()
        frame.addView(video, FrameLayout.LayoutParams(-1, -1))
        val overlay = column().apply { setPadding(px(36), px(18), px(36), px(18)); setBackgroundColor(0xDD101913.toInt()) }
        fullOverlay = overlay
        frame.addView(overlay, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        val clip = fullClip
        val camera = if (clip == null) cameras.getOrNull(selected) else null
        if (clip == null && camera == null) { loadCameras(); return }
        overlay.addView(text(if (clip == null) camera!!.name else "${formatTime(clip.time)}  /  ${clip.caption}", 22f, true).apply {
            maxLines = 2; ellipsize = android.text.TextUtils.TruncateAt.END
        })
        val state = note(overlay, "Connecting...")
        fullStatus = state
        val controls = row()
        val back = button(if (clip == null) "Camera grid" else "Replays") { goBack() }; controls.addView(back)
        fullBack = back
        lateinit var toggle: Button
        toggle = button("Pause") {
            slots.firstOrNull()?.player?.let { it.playWhenReady = !it.playWhenReady; toggle.text = if (it.playWhenReady) "Pause" else "Play" }
        }
        controls.addView(toggle)
        playButton = toggle
        if (clip != null) {
            controls.addView(button("Back 10s") { seek(-10_000) })
            controls.addView(button("Forward 10s") { seek(10_000) })
        } else controls.addView(button("Go live") { slots.firstOrNull()?.player?.let { it.seekToDefaultPosition(); it.play() } })
        lateinit var audio: Button
        audio = button(if (sound) "Sound on" else "Sound off") { sound = !sound; audio.text = if (sound) "Sound on" else "Sound off"; applyAudio() }
        controls.addView(audio)
        controls.addView(button("Retry") { showFull() })
        overlay.addView(controls)
        if (clip != null) note(overlay, "Remote left / right moves between controls. Rewind / fast-forward skips 10 seconds.")
        val slot = Slot(selected, camera, video, state, null)
        slots += slot
        if (clip == null) slot.startLive() else slot.start(clip.video, false)
        back.requestFocus()
        handler.postDelayed(hideControls, 6000)
        if (clip != null) updateReplayClock()
    }

    private fun updateReplayClock() {
        if (mode != "full" || fullClip == null) return
        slots.firstOrNull()?.player?.let { player ->
            if (player.playbackState == Player.STATE_READY) {
                fun clock(ms: Long): String = "%d:%02d".format(ms.coerceAtLeast(0) / 60000, (ms.coerceAtLeast(0) / 1000) % 60)
                fullStatus?.text = "${if (player.playWhenReady) "Playing" else "Paused"}  ${clock(player.currentPosition)} / ${clock(player.duration)}"
            }
        }
        handler.postDelayed({ updateReplayClock() }, 1000)
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val navigation = event.keyCode in listOf(KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER,
            KeyEvent.KEYCODE_DPAD_UP, KeyEvent.KEYCODE_DPAD_DOWN, KeyEvent.KEYCODE_DPAD_LEFT, KeyEvent.KEYCODE_DPAD_RIGHT)
        if (mode == "grid") {
            if (event.keyCode == KeyEvent.KEYCODE_MENU) {
                if (event.action == KeyEvent.ACTION_DOWN) showWallControls()
                return true
            }
            if (navigation && wallOverlay?.visibility == View.VISIBLE) {
                handler.removeCallbacks(hideWallControls); handler.postDelayed(hideWallControls, 6000)
            }
        }
        if (mode == "full" && navigation) {
            if (fullOverlay?.visibility == View.GONE) {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    fullOverlay?.visibility = View.VISIBLE
                    fullBack?.requestFocus()
                    handler.removeCallbacks(hideControls); handler.postDelayed(hideControls, 6000)
                }
                return true
            }
            handler.removeCallbacks(hideControls); handler.postDelayed(hideControls, 6000)
        }
        return super.dispatchKeyEvent(event)
    }

    private fun seek(delta: Long) {
        slots.firstOrNull()?.player?.let { player ->
            val end = if (player.duration == C.TIME_UNSET) Long.MAX_VALUE else player.duration
            player.seekTo((player.currentPosition + delta).coerceIn(0, end))
        }
    }
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (mode == "full") when (keyCode) {
            KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE -> { slots.firstOrNull()?.player?.let { it.playWhenReady = !it.playWhenReady }; return true }
            KeyEvent.KEYCODE_MEDIA_PLAY -> { slots.firstOrNull()?.player?.play(); return true }
            KeyEvent.KEYCODE_MEDIA_PAUSE -> { slots.firstOrNull()?.player?.pause(); return true }
            KeyEvent.KEYCODE_MEDIA_REWIND -> { if (fullClip != null) seek(-10_000); return true }
            KeyEvent.KEYCODE_MEDIA_FAST_FORWARD -> { if (fullClip != null) seek(10_000); return true }
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun playerView() = (layoutInflater.inflate(R.layout.player, null) as PlayerView).apply {
        isFocusable = false; descendantFocusability = ViewGroup.FOCUS_BLOCK_DESCENDANTS
        useController = false
    }

    private inner class Slot(val index: Int, val camera: Camera?, val view: PlayerView, val state: TextView, val image: ImageView?) {
        var player: ExoPlayer? = null
        var snapshotBusy = false
        private var retries = 0
        private var released = false
        private var address = ""
        private var live = true
        private var bufferingSince = 0L
        private var retryPending = false
        private val ticket = generation
        private val watchdog = object : Runnable {
            override fun run() {
                if (released || ticket != generation || player == null) return
                val current = player!!
                if (current.playWhenReady && current.playbackState == Player.STATE_BUFFERING) {
                    if (bufferingSince == 0L) bufferingSince = android.os.SystemClock.elapsedRealtime()
                    if (android.os.SystemClock.elapsedRealtime() - bufferingSince > 30_000) { bufferingSince = 0; retry() }
                } else bufferingSince = 0
                handler.postDelayed(this, 5000)
            }
        }
        fun startLive() { camera?.let { start(api.url("/api/cameras/${android.net.Uri.encode(it.id)}/hls/live.m3u8"), true) } }
        fun start(url: String, isLive: Boolean) {
            player?.release(); released = false; address = url; live = isLive
            view.visibility = View.VISIBLE
            val factory = DefaultHttpDataSource.Factory().setConnectTimeoutMs(8000).setReadTimeoutMs(12000)
                .setDefaultRequestProperties(mapOf("Cookie" to api.cookie)).setUserAgent("AviaryTV/0.1")
            val source = DefaultMediaSourceFactory(factory)
            val instance = ExoPlayer.Builder(this@MainActivity).setMediaSourceFactory(source)
                .setLoadControl(DefaultLoadControl.Builder().setBufferDurationsMs(2000, 10000, 1000, 1500).build()).build()
            player = instance; view.player = instance
            // A single selected stream has audible volume. Other grid decoders stay muted.
            instance.setAudioAttributes(AudioAttributes.Builder().setUsage(C.USAGE_MEDIA).setContentType(C.AUDIO_CONTENT_TYPE_MOVIE).build(), false)
            instance.addListener(object : Player.Listener {
                override fun onPlayWhenReadyChanged(playWhenReady: Boolean, reason: Int) {
                    if (!released && mode == "full") playButton?.text = if (playWhenReady) "Pause" else "Play"
                }
                override fun onPlaybackStateChanged(playbackState: Int) {
                    if (released) return
                    state.visibility = if (mode == "grid" && playbackState == Player.STATE_READY) View.INVISIBLE else View.VISIBLE
                    state.text = when (playbackState) {
                        Player.STATE_READY -> { retries = 0; if (live) "LIVE" else "Replay" }
                        Player.STATE_ENDED -> if (live) "Reconnecting..." else "Replay finished"
                        Player.STATE_BUFFERING -> "Buffering..."
                        else -> "Connecting..."
                    }
                    if (live && playbackState == Player.STATE_ENDED) retry()
                }
                override fun onPlayerError(error: PlaybackException) {
                    if (released) return
                    var cause: Throwable? = error
                    while (cause != null) {
                        if (cause is HttpDataSource.InvalidResponseCodeException && cause.responseCode in listOf(401, 403)) {
                            handler.post { if (ticket == generation) showError(LoginRequired()) }; return
                        }
                        cause = cause.cause
                    }
                    if (mode == "grid" && error.errorCode in 4000..4999) {
                        handler.post { if (ticket == generation) { preview(); status?.text = "This TV reached its decoder limit. Preview tiles refresh every 5 seconds; select for full screen." } }
                    } else retry()
                }
            })
            prepare()
            if (!live && replayPosition > 0) instance.seekTo(replayPosition)
            instance.playWhenReady = live || replayPlaying
            applyAudio()
            handler.removeCallbacks(watchdog); handler.postDelayed(watchdog, 5000)
        }
        private fun prepare() {
            player?.setMediaItem(MediaItem.Builder().setUri(address).setMimeType(if (live) MimeTypes.APPLICATION_M3U8 else MimeTypes.VIDEO_MP4).build())
            player?.prepare()
        }
        private fun retry() {
            if (retryPending || released || player == null) return
            retryPending = true
            val delay = (1000L shl retries.coerceAtMost(4)).coerceAtMost(15000)
            retries++
            state.text = "Reconnecting in ${delay / 1000}s..."
            val position = if (live) 0L else player?.currentPosition ?: 0L
            handler.postDelayed({
                retryPending = false
                if (!released && ticket == generation && started) {
                    prepare()
                    if (!live) player?.seekTo(position)
                }
            }, delay)
        }
        fun preview() {
            player?.let { view.player = null; it.release() }; player = null
            handler.removeCallbacks(watchdog); view.visibility = View.INVISIBLE
            state.text = "Preview / 5s"
            state.visibility = View.VISIBLE
        }
        fun release() {
            released = true; handler.removeCallbacks(watchdog)
            view.player = null; player?.release(); player = null
        }
    }
}
