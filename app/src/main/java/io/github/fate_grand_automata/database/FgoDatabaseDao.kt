/**
 * This file defines the Data Access Object (DAO) for the FGO-Bot database.
 * The DAO provides an abstract interface for performing database operations.
 */
package io.github.fate_grand_automata.database

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/**
 * The Data Access Object for interacting with the user's Servant and CE database.
 * This interface is used by Room to generate the necessary code for database operations.
 */
@Dao
interface FgoDatabaseDao {

    // Servant Queries
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertServant(servant: UserServant)

    @Update
    suspend fun updateServant(servant: UserServant)

    @Query("SELECT * FROM user_servants ORDER BY gameId ASC")
    fun getAllServants(): Flow<List<UserServant>>

    @Query("SELECT * FROM user_servants WHERE id = :id")
    suspend fun getServant(id: Int): UserServant?

    @Query("DELETE FROM user_servants WHERE id = :id")
    suspend fun deleteServant(id: Int)

    // Craft Essence Queries
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCraftEssence(ce: UserCraftEssence)

    @Update
    suspend fun updateCraftEssence(ce: UserCraftEssence)

    @Query("SELECT * FROM user_craft_essences ORDER BY name ASC")
    fun getAllCraftEssences(): Flow<List<UserCraftEssence>>

    @Query("DELETE FROM user_craft_essences WHERE id = :id")
    suspend fun deleteCraftEssence(id: Int)
} 