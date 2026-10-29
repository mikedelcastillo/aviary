package org.aviary.tv

internal object AppPins {
    fun visible(saved: List<String>?, available: List<String>): List<String> =
        (saved ?: available.take(8)).distinct().filter { it in available }

    fun select(existing: List<String>, selected: List<String>): List<String> =
        existing.filter { it in selected } + selected.filter { it !in existing }.distinct()

    fun move(existing: List<String>, item: String, offset: Int): List<String> {
        val index = existing.indexOf(item)
        if (index < 0) return existing
        val target = (index + offset).coerceIn(existing.indices)
        return existing.toMutableList().apply { removeAt(index); add(target, item) }
    }
}
