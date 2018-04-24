package io.improbable.keanu.util.csv.pojo;

import java.util.function.BiConsumer;

/**
 * A ColumnConsumer is a function that takes some object
 * and some String value and applies the String value to the
 * object. For example, if a given target object class type has
 * a public field named "something" that mapped to a csv column
 * header named "something" then this CsvColumnConsumer would set
 * the public field of an instance of (T) to the String value also
 * provided. Every column in the csv source should map to a type
 * of CsvColumnConsumer, whether it's a public field CsvColumnConsumer
 * or a setter method CsvColumnConsumer.
 *
 * @param <T> target object type
 */
public interface CsvColumnConsumer<T> extends BiConsumer<T, String> {
}
