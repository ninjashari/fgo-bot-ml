/**
 * This file defines the data structure for storing user-owned Servants
 * in the local Room database.
 */
package io.github.fate_grand_automata.database

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Represents a single Servant owned by the user. This entity is used by Room
 * to create the 'user_servants' table in the local database.
 *
 * @property id The unique identifier for this servant entry in the database.
 * @property gameId The in-game identifier of the Servant (e.g., from Atlas Academy).
 * @property name The name of the Servant.
 * @property level The current level of the Servant.
 * @property ascension The current ascension level.
 * @property npLevel The Noble Phantasm level.
 * @property skill1Level The level of the first skill.
 * @property skill2Level The level of the second skill.
 * @property skill3Level The level of the third skill.
 * @property servantClass The class of the Servant.
 * @property cost The cost of the Servant.
 * @property atk The attack power of the Servant.
 * @property hp The health points of the Servant.
 * @property appendSkill1Level The level of the first append skill.
 * @property appendSkill2Level The level of the second append skill.
 * @property appendSkill3Level The level of the third append skill.
 * @property bondLevel The bond level of the Servant.
 * @property gender The gender of the Servant.
 * @property alignmentsList The list of alignments of the Servant.
 * @property traitsList The list of traits of the Servant.
 */
@Entity(tableName = "user_servants")
data class UserServant(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val gameId: Int,
    val name: String,
    val level: Int,
    val ascension: Int,
    val npLevel: Int,
    val skill1Level: Int,
    val skill2Level: Int,
    val skill3Level: Int,
    val servantClass: String,
    val cost: Int,
    val atk: Int,
    val hp: Int,
    val appendSkill1Level: Int = 0,
    val appendSkill2Level: Int = 0,
    val appendSkill3Level: Int = 0,
    val bondLevel: Int,
    val gender: String,
    val alignmentsList: List<String> = emptyList(),
    val traitsList: List<String> = emptyList()
) 