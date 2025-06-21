/**
 * This file defines the main Room Database class for the application.
 * It serves as the primary access point to the persisted data.
 */
package io.github.fate_grand_automata.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

/**
 * The main database class for the application. This class is annotated with @Database
 * and lists all the entities and the database version. It provides access to the DAO.
 */
@Database(entities = [UserServant::class, UserCraftEssence::class], version = 2, exportSchema = false)
@TypeConverters(TypeConverters::class)
abstract class FgoDatabase : RoomDatabase() {

    abstract fun fgoDatabaseDao(): FgoDatabaseDao

    companion object {
        @Volatile
        private var INSTANCE: FgoDatabase? = null

        /**
         * Returns an instance of the database. Implements a singleton pattern to prevent
         * multiple instances of the database from being opened at the same time.
         *
         * @param context The application context.
         * @return The singleton instance of the FgoDatabase.
         */
        fun getInstance(context: Context): FgoDatabase {
            synchronized(this) {
                var instance = INSTANCE

                if (instance == null) {
                    instance = Room.databaseBuilder(
                        context.applicationContext,
                        FgoDatabase::class.java,
                        "fgo_database"
                    )
                        .fallbackToDestructiveMigration()
                        .build()
                    INSTANCE = instance
                }
                return instance
            }
        }
    }
} 