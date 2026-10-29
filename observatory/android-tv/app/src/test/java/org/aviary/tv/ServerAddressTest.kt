package org.aviary.tv

import org.junit.Assert.*
import org.junit.Test

class ServerAddressTest {
    @Test fun localAndSecureAddresses() {
        assertEquals("http://192.168.10.8:8740", ServerAddress.normalize(" http://192.168.10.8:8740/ "))
        assertEquals("https://aviary.example", ServerAddress.normalize("https://aviary.example"))
        assertEquals("http://172.16.0.8", ServerAddress.normalize("http://172.16.0.8"))
    }
    @Test fun refusesInsecureRemoteOrEmbeddedCredentials() {
        for (address in listOf("http://example.com", "http://172.32.0.1", "ftp://192.168.0.2", "http://user:pin@192.168.0.2", "http://192.168.0.2/path", "http://192.168.0.2?pin=a")) {
            assertThrows(IllegalArgumentException::class.java) { ServerAddress.normalize(address) }
        }
    }
    @Test fun sessionStaysOnServerOrigin() {
        val base = "https://aviary.example"
        assertEquals("$base/api/clips/test/video", ServerAddress.resolve(base, "/api/clips/test/video"))
        for (path in listOf("//other.example/video", "https://other.example/video", "http://aviary.example/video", "https://aviary.example:80/video")) {
            assertThrows(IllegalArgumentException::class.java) { ServerAddress.resolve(base, path) }
        }
    }
}
