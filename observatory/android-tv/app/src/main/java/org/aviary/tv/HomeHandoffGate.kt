package org.aviary.tv

/** Only the stock Google TV Home activity can request a handoff. No event content is kept. */
internal class HomeHandoffGate {
    private var connected = false
    private var lastLaunch: Long? = null

    fun connect() { connected = true; lastLaunch = null }
    fun disconnect() { connected = false; lastLaunch = null }

    fun accept(packageName: CharSequence?, className: CharSequence?, eventTime: Long, now: Long): Boolean {
        if (!connected || packageName?.toString() != STOCK_PACKAGE || className?.toString() != STOCK_ACTIVITY) return false
        if (now - eventTime !in 0..2000) return false
        if (lastLaunch?.let { now - it < 1500 } == true) return false
        lastLaunch = now
        return true
    }

    companion object {
        const val STOCK_PACKAGE = "com.google.android.apps.tv.launcherx"
        const val STOCK_ACTIVITY = "com.google.android.apps.tv.launcherx.home.HomeActivity"
    }
}
