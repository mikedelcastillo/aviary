package org.aviary.tv

import org.junit.Assert.*
import org.junit.Test

class HomeHandoffGateTest {
    private val stockPackage = HomeHandoffGate.STOCK_PACKAGE
    private val stockActivity = HomeHandoffGate.STOCK_ACTIVITY

    @Test fun onlyExactStockHomeMatches() {
        val gate = HomeHandoffGate().apply { connect() }
        for ((pkg, activity) in listOf(
            "org.aviary.tv" to "org.aviary.tv.MainActivity",
            "com.android.tv.settings" to "com.android.tv.settings.MainSettings",
            stockPackage to "$stockPackage.settings.SettingsActivity",
            "com.example.other" to stockActivity,
            stockPackage to ".home.HomeActivity"
        )) assertFalse(gate.accept(pkg, activity, 10000, 10000))
        assertFalse(gate.accept(null, null, 10000, 10000))
        assertTrue(gate.accept(stockPackage, stockActivity, 10000, 10000))
    }

    @Test fun rejectsStaleAndFutureEvents() {
        val gate = HomeHandoffGate().apply { connect() }
        assertFalse(gate.accept(stockPackage, stockActivity, 7000, 10000))
        assertFalse(gate.accept(stockPackage, stockActivity, 10001, 10000))
        assertTrue(gate.accept(stockPackage, stockActivity, 9900, 10000))
    }

    @Test fun duplicatesAreIgnoredButRapidRealHomePressesStillWork() {
        val gate = HomeHandoffGate().apply { connect() }
        assertTrue(gate.accept(stockPackage, stockActivity, 10000, 10000))
        assertFalse(gate.accept(stockPackage, stockActivity, 10000, 10010))
        assertFalse(gate.accept(stockPackage, stockActivity, 9999, 10010))
        assertTrue(gate.accept(stockPackage, stockActivity, 10020, 10030))
    }

    @Test fun disconnectedLifecycleCannotLaunch() {
        val gate = HomeHandoffGate()
        assertFalse(gate.accept(stockPackage, stockActivity, 10000, 10000))
        gate.connect()
        assertTrue(gate.accept(stockPackage, stockActivity, 10000, 10000))
        gate.disconnect()
        assertFalse(gate.accept(stockPackage, stockActivity, 12000, 12000))
        gate.connect()
        assertTrue(gate.accept(stockPackage, stockActivity, 13000, 13000))
    }
}
