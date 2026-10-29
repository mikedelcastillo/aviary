package org.aviary.tv

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HomeHandoffTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private class Harness(context: Context) : HomeHandoffService() {
        val launched = mutableListOf<Intent>()
        var denyLaunch = false
        init { attachBaseContext(context) }
        fun connect() { onServiceConnected() }
        override fun startActivity(intent: Intent) {
            if (denyLaunch) throw SecurityException("Synthetic firmware launch restriction")
            launched += intent
        }
    }
    @Suppress("DEPRECATION")
    private fun event(pkg: String = HomeHandoffGate.STOCK_PACKAGE, activity: String = HomeHandoffGate.STOCK_ACTIVITY,
                      type: Int = AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) = AccessibilityEvent.obtain(type).apply {
        packageName = pkg; className = activity; eventTime = SystemClock.uptimeMillis()
    }

    @Test fun fabricatedEventsRespectFilterLifecycleAndHomeIntentWithoutPermission() {
        val service = Harness(context)
        service.onAccessibilityEvent(event())
        assertTrue(service.launched.isEmpty())
        service.connect()
        service.onAccessibilityEvent(event(type = AccessibilityEvent.TYPE_VIEW_CLICKED))
        service.onAccessibilityEvent(event("org.aviary.tv", MainActivity::class.java.name))
        service.onAccessibilityEvent(event("com.android.tv.settings", "com.android.tv.settings.MainSettings"))
        service.onAccessibilityEvent(event(activity = "android.widget.FrameLayout"))
        assertTrue(service.launched.isEmpty())
        service.onAccessibilityEvent(event())
        val launch = service.launched.single()
        assertEquals(Intent.ACTION_MAIN, launch.action)
        assertTrue(launch.hasCategory(Intent.CATEGORY_HOME))
        assertEquals(MainActivity::class.java.name, launch.component?.className)
        assertEquals(context.packageName, launch.component?.packageName)
        assertTrue(launch.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)
        service.onAccessibilityEvent(event())
        assertEquals(1, service.launched.size)
        service.onUnbind(Intent())
        service.onAccessibilityEvent(event())
        assertEquals(1, service.launched.size)
        service.connect()
        service.onAccessibilityEvent(event())
        assertEquals(2, service.launched.size)
        service.onDestroy()
        service.onAccessibilityEvent(event())
        assertEquals(2, service.launched.size)
    }

    @Test fun firmwareLaunchDenialDoesNotCrashOrRetryLoop() {
        val service = Harness(context).apply { connect(); denyLaunch = true }
        service.onAccessibilityEvent(event())
        service.denyLaunch = false
        service.onAccessibilityEvent(event())
        assertTrue(service.launched.isEmpty())
        service.onDestroy()
    }

    @Test fun manifestScopeHasNoContentKeyOrGestureCapabilities() {
        val manager = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
        val info = manager.installedAccessibilityServiceList.single {
            it.resolveInfo.serviceInfo.packageName == context.packageName && it.resolveInfo.serviceInfo.name == HomeHandoffService::class.java.name
        }
        assertEquals(AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED, info.eventTypes)
        assertArrayEquals(arrayOf(HomeHandoffGate.STOCK_PACKAGE), info.packageNames)
        assertEquals(0, info.capabilities)
        assertEquals(0, info.flags)
        assertEquals("android.permission.BIND_ACCESSIBILITY_SERVICE", info.resolveInfo.serviceInfo.permission)
        // These fixture tests never enable the service or change global settings.
    }
}
