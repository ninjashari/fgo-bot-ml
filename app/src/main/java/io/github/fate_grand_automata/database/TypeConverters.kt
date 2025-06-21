/**
 * This file contains TypeConverters for the Room database. These converters
 * allow Room to persist complex data types that it doesn't natively support.
 */
package io.github.fate_grand_automata.database

import androidx.room.TypeConverter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * Provides methods to convert complex types to and from a format that the
 * Room database can persist. This class uses Gson for JSON serialization.
 */
class TypeConverters {
    private val gson = Gson()

    /**
     * Converts a JSON string back into a List of Strings.
     */
    @TypeConverter
    fun fromStringList(value: String?): List<String> {
        if (value == null) {
            return emptyList()
        }
        val listType = object : TypeToken<List<String>>() {}.type
        return gson.fromJson(value, listType)
    }

    /**
     * Converts a List of Strings into a JSON string for database storage.
     */
    @TypeConverter
    fun fromList(list: List<String>?): String {
        return gson.toJson(list)
    }

    /**
     * Converts a JSON string back into a List of Effect objects.
     */
    @TypeConverter
    fun fromEffectListString(value: String?): List<Effect> {
        if (value == null) {
            return emptyList()
        }
        val listType = object : TypeToken<List<Effect>>() {}.type
        return gson.fromJson(value, listType)
    }

    /**
     * Converts a List of Effect objects into a JSON string for database storage.
     */
    @TypeConverter
    fun fromEffectList(list: List<Effect>?): String {
        return gson.toJson(list)
    }
} 