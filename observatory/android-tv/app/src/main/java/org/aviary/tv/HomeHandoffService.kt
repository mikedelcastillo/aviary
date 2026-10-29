package org.aviary.tv

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent

/** Optional owner-enabled Home handoff. Never reads windows, nodes, text or event contents. */
open class HomeHandoffService : AccessibilityService() {
    private val gate = HomeHandoffGate()

    override fun onServiceConnected() { gate.connect() }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null || event.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) return
        if (!gate.accept(event.packageName, event.className, event.eventTime, SystemClock.uptimeMillis())) return
        val home = Intent(this, MainActivity::class.java)
            .setAction(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        // A firmware launch restriction must leave the system Home usable.
        runCatching { startActivity(home) }
    }

    override fun onInterrupt() { /* No feedback, queued work or delayed launches. */ }
    override fun onUnbind(intent: Intent?): Boolean { gate.disconnect(); return super.onUnbind(intent) }
    override fun onDestroy() { gate.disconnect(); super.onDestroy() }
}
