package org.aviary.tv

import java.net.URI

/** Keep the session on its configured origin, including server-provided media links. */
internal object ServerAddress {
    fun normalize(raw: String): String {
        val value = raw.trim().trimEnd('/')
        val uri = try { URI(value) } catch (_: Exception) { throw IllegalArgumentException("Enter a full server URL.") }
        require(uri.scheme == "http" || uri.scheme == "https") { "Use http:// or https://." }
        require(!uri.host.isNullOrBlank() && uri.userInfo == null && uri.query == null && uri.fragment == null && uri.path.isNullOrEmpty()) {
            "Use the server address and port, without a path or password."
        }
        require(uri.port == -1 || uri.port in 1..65535) { "The port must be between 1 and 65535." }
        require(uri.scheme == "https" || isLocalHost(uri.host)) { "HTTP is only allowed for a local network address. Use HTTPS remotely." }
        return value
    }

    private fun isLocalHost(host: String): Boolean {
        if (host == "localhost" || host.endsWith(".local") || host == "[::1]") return true
        val ipv6 = host.removeSurrounding("[", "]").lowercase()
        if (ipv6.startsWith("fe80:") || Regex("f[cd][0-9a-f]{2}:.*").matches(ipv6)) return true
        val parts = host.split('.').map { it.toIntOrNull() ?: return false }
        if (parts.size != 4 || parts.any { it !in 0..255 }) return false
        return parts[0] == 10 || parts[0] == 127 ||
            (parts[0] == 192 && parts[1] == 168) || (parts[0] == 172 && parts[1] in 16..31)
    }

    fun resolve(base: String, path: String): String {
        val root = URI(base)
        val resolved = root.resolve(path)
        require(resolved.scheme == root.scheme && resolved.host == root.host && resolved.port == root.port && resolved.userInfo == null) {
            "The server returned a media URL outside this server."
        }
        return resolved.toString()
    }
}
