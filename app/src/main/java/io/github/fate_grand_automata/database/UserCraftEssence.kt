/**
 * This file defines the data structure for storing user-owned Craft Essences
 * in the local Room database.
 */
package io.github.fate_grand_automata.database

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Represents a single Craft Essence owned by the user. This entity is used by Room
 * to create the 'user_craft_essences' table in the local database.
 *
 * @property id The unique identifier for this CE entry in the database.
 * @property gameId The in-game identifier of the Craft Essence.
 * @property name The name of the Craft Essence.
 * @property level The current level of the Craft Essence.
 * @property mlb True if the Craft Essence is Max Limit Broken, false otherwise.
 * @property atk The attack power of the Craft Essence.
 * @property hp The HP of the Craft Essence.
 * @property cost The cost of the Craft Essence.
 * @property effectsList The list of effects associated with the Craft Essence.
 */
@Entity(tableName = "user_craft_essences")
data class UserCraftEssence(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val gameId: Int,
    val name: String,
    val level: Int,
    val mlb: Boolean,
    val atk: Int,
    val hp: Int,
    val cost: Int,
    val effectsList: List<Effect> = emptyList()
) 