package org.aviary.tv

import org.junit.Assert.assertEquals
import org.junit.Test

class AppPinsTest {
    @Test fun hiddenAndUninstalledAppsStayOutOfTheRow() {
        assertEquals(listOf("b", "a"), AppPins.visible(listOf("b", "gone", "a"), listOf("a", "b", "c")))
        assertEquals(emptyList<String>(), AppPins.visible(emptyList(), listOf("a", "b")))
    }
    @Test fun selectionPreservesExistingOrderAndAppendsNewPins() {
        assertEquals(listOf("b", "c", "d"), AppPins.select(listOf("b", "a", "c"), listOf("c", "b", "d")))
    }
    @Test fun movingDoesNotLosePinsAtEitherBoundary() {
        assertEquals(listOf("b", "a", "c"), AppPins.move(listOf("a", "b", "c"), "b", -1))
        assertEquals(listOf("a", "b", "c"), AppPins.move(listOf("a", "b", "c"), "a", -1))
        assertEquals(listOf("b", "c", "a"), AppPins.move(listOf("a", "b", "c"), "a", 10))
    }
}
