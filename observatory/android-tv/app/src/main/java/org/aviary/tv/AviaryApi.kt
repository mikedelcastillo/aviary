package org.aviary.tv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

internal class LoginRequired : IOException("Your session expired. Enter your passcode again.")
internal data class Camera(val id: String, val name: String, val online: Boolean)
internal data class Clip(val id: String, val time: String, val camera: String, val duration: Double, val video: String, val caption: String)
internal data class ClipPage(val clips: List<Clip>, val next: String?)

internal class AviaryApi(context: Context) {
    private val prefs = context.getSharedPreferences("connection", Context.MODE_PRIVATE)
    @Volatile var base = prefs.getString("server", "") ?: ""
        private set
    @Volatile var cookie = prefs.getString("session", "") ?: ""
        private set
    val connected get() = base.isNotEmpty() && cookie.isNotEmpty()
    fun url(path: String) = ServerAddress.resolve(base, path)

    private fun connection(address: String, session: String = cookie): HttpURLConnection = (URL(address).openConnection() as HttpURLConnection).apply {
        connectTimeout = 8000
        readTimeout = 12000
        instanceFollowRedirects = false
        setRequestProperty("Accept", "application/json")
        if (session.isNotEmpty()) setRequestProperty("Cookie", session)
    }

    fun login(server: String, pin: String) {
        val normalized = ServerAddress.normalize(server)
        val request = connection("$normalized/api/login", "")
        try {
            request.requestMethod = "POST"
            request.doOutput = true
            request.setRequestProperty("Content-Type", "application/json")
            request.outputStream.use { it.write(JSONObject().put("pin", pin).toString().toByteArray()) }
            if (request.responseCode in listOf(401, 403)) throw IOException("That passcode was not accepted.")
            if (request.responseCode == 429) throw IOException("Too many attempts. Wait a minute, then try again.")
            if (request.responseCode !in 200..299) throw IOException("The server could not sign in (${request.responseCode}).")
            val session = request.headerFields.entries.filter { it.key.equals("Set-Cookie", true) }.flatMap { it.value }
                .map { it.substringBefore(';') }.firstOrNull { it.startsWith("aviary_session=") && it.length > 15 }
                ?: throw IOException("The server did not return a session.")
            base = normalized
            cookie = session
            prefs.edit().putString("server", base).putString("session", cookie).apply()
        } finally { request.disconnect() }
    }

    fun signOut() { cookie = ""; prefs.edit().remove("session").apply() }

    fun verifyDiscovery(server: String): String {
        val address = ServerAddress.normalize(server)
        val request = connection("$address/api/discovery", "")
        request.connectTimeout = 3000; request.readTimeout = 3000
        try {
            if (request.responseCode != 200) throw IOException("Discovery identity unavailable")
            val identity = request.inputStream.bufferedReader().use { reader ->
                val buffer = CharArray(4097)
                var count = 0
                while (count < buffer.size) {
                    val read = reader.read(buffer, count, buffer.size - count)
                    if (read < 0) break
                    count += read
                }
                if (count > 4096) throw IOException("Discovery identity is too large")
                JSONObject(String(buffer, 0, count))
            }
            if (identity.optString("service") != "aviary-observatory" || identity.optInt("api") != 1) throw IOException("This is not an Aviary server")
            return address
        } finally { request.disconnect() }
    }

    private fun json(path: String): JSONObject {
        val request = connection(url(path))
        try {
            if (request.responseCode in listOf(401, 403)) throw LoginRequired()
            if (request.responseCode !in 200..299) throw IOException("Server unavailable (${request.responseCode}). Try again.")
            return request.inputStream.bufferedReader().use { JSONObject(it.readText()) }
        } finally { request.disconnect() }
    }

    fun cameras(): List<Camera> {
        val data = json("/api/overview").getJSONArray("cameras")
        return (0 until data.length()).map { index -> data.getJSONObject(index).let {
            Camera(it.getString("id"), it.optString("name", "Camera"), it.optBoolean("online"))
        } }
    }

    fun clips(view: String, cursor: String?): ClipPage {
        val query = if (cursor == null) "" else "&before=" + java.net.URLEncoder.encode(cursor, "UTF-8")
        val data = json("/api/clips?view=$view&limit=24$query")
        val rows = data.getJSONArray("clips")
        val clips = (0 until rows.length()).map { index -> rows.getJSONObject(index).let {
            Clip(it.getString("id"), it.getString("time"), it.optString("camera"), it.optDouble("duration", 0.0),
                url(it.getString("video")), it.optString("caption").takeIf { c -> c.isNotBlank() } ?: "Recording")
        } }
        return ClipPage(clips, data.optString("next").takeIf { it.isNotBlank() && it != "null" })
    }

    fun snapshot(camera: Camera): Bitmap? {
        val request = connection(url("/api/cameras/${android.net.Uri.encode(camera.id)}/snapshot"))
        try {
            if (request.responseCode in listOf(401, 403)) throw LoginRequired()
            if (request.responseCode != 200) return null
            return request.inputStream.use { BitmapFactory.decodeStream(it) }
        } finally { request.disconnect() }
    }
}
