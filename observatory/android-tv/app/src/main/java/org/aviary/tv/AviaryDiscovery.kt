package org.aviary.tv

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper

/** Browse only Aviary's advertised service; no subnet or port scanning. */
internal class AviaryDiscovery(context: Context, private val found: (String) -> Unit, private val unavailable: () -> Unit) {
    private val nsd = context.getSystemService(Context.NSD_SERVICE) as NsdManager
    private val handler = Handler(Looper.getMainLooper())
    private var active = false
    private var resolving = false
    private val pending = ArrayDeque<NsdServiceInfo>()
    private val seen = mutableSetOf<String>()
    private val multicast = (context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager)
        .createMulticastLock("aviary-discovery").apply { setReferenceCounted(false) }
    private val listener = object : NsdManager.DiscoveryListener {
        override fun onDiscoveryStarted(type: String) = Unit
        override fun onDiscoveryStopped(type: String) = Unit
        override fun onServiceLost(service: NsdServiceInfo) = Unit
        override fun onStartDiscoveryFailed(type: String, code: Int) { handler.post { if (active) { stop(); unavailable() } } }
        override fun onStopDiscoveryFailed(type: String, code: Int) = Unit
        override fun onServiceFound(service: NsdServiceInfo) {
            handler.post {
                if (active && service.serviceType.trimEnd('.') == "_aviary._tcp" && seen.add(service.serviceName)) {
                    pending.add(service); resolveNext()
                }
            }
        }
    }
    fun start() {
        active = true
        runCatching { multicast.acquire() }
        try { nsd.discoverServices("_aviary._tcp.", NsdManager.PROTOCOL_DNS_SD, listener) }
        catch (_: RuntimeException) { stop(); unavailable(); return }
        handler.postDelayed({ if (active) { stop(); unavailable() } }, 10000)
    }
    @Suppress("DEPRECATION")
    private fun resolveNext() {
        if (!active || resolving || pending.isEmpty()) return
        resolving = true
        nsd.resolveService(pending.removeFirst(), object : NsdManager.ResolveListener {
            override fun onResolveFailed(info: NsdServiceInfo, code: Int) { handler.post { resolving = false; resolveNext() } }
            override fun onServiceResolved(info: NsdServiceInfo) {
                handler.post {
                    resolving = false
                    if (!active) return@post
                    val scheme = info.attributes["scheme"]?.toString(Charsets.UTF_8) ?: "http"
                    val api = info.attributes["api"]?.toString(Charsets.UTF_8)
                    val host = info.host?.hostAddress
                    if (api == "1" && scheme in listOf("http", "https") && host != null && info.port in 1..65535) {
                        val literal = if (host.contains(':')) "[$host]" else host
                        found("$scheme://$literal:${info.port}")
                    }
                    resolveNext()
                }
            }
        })
    }
    fun stop() {
        if (!active) return
        active = false
        handler.removeCallbacksAndMessages(null)
        runCatching { nsd.stopServiceDiscovery(listener) }
        if (multicast.isHeld) multicast.release()
        pending.clear()
    }
}
