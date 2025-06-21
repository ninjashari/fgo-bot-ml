/**
 * This file defines the data structure for a Craft Essence effect.
 */
package io.github.fate_grand_automata.database

/**
 * Represents a single effect of a Craft Essence.
 * This is not a table in the database, but is used as a complex type within
 * the UserCraftEssence entity.
 *
 * @property effect The description of the effect (e.g., "NP Strength Up").
 * @property value The value or magnitude of the effect (e.g., "15%").
 */
data class Effect(
    val effect: String,
    val value: String
) 