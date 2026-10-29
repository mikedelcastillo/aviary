package org.aviary.tv

/** Only the stock Google TV Home activity can request a handoff. No event content is kept. */
internal class HomeHandoffGate {
    private var connected = false
    private var lastEventTime: Long? = null

    fun connect() { connected = true; lastEventTime = null }
    fun disconnect() { connected = false; lastEventTime = null }

    fun accept(packageName: CharSequence?, className: CharSequence?, eventTime: Long, now: Long): Boolean {
        if (!connected || packageName?.toString() != STOCK_PACKAGE || className?.toString() != STOCK_ACTIVITY) return false
        if (now - eventTime !in 0..2000) return false
        // Only discard duplicate or older deliveries: a time cooldown can swallow
        // a real second Home press and strand the user in the stock launcher.
        // Framework event coalescing is also disabled, because a generic window
        // event can otherwise replace the exact Home activity event.
        if (lastEventTime?.let { eventTime <= it } == true) return false
        lastEventTime = eventTime
        return true
    }

    companion object {
        const val STOCK_PACKAGE = "com.google.android.apps.tv.launcherx"
        const val STOCK_ACTIVITY = "com.google.android.apps.tv.launcherx.home.HomeActivity"
    }
}
