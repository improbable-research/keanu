package io.improbable.keanu.util.csv.pojo.byrow;

import java.util.function.BiConsumer;

/**
 * A CsvCellConsumer is a function that takes some object
 * and some String value and applies the String value to the
 * object. For example, if a given target object class type has
 * a public field named "something" that mapped to a csv column
 * header named "something" then this CsvCellConsumer would set
 * the public field of an instance of (T) to the String value also
 * provided. Every column in the csv source should map to a type
 * of CsvCellConsumer, whether it's a public field CsvColumnConsumer
 * or a setter method CsvColumnConsumer.
 *
 * @param <T> target object type
 */
public interface CsvCellConsumer<T> extends BiConsumer<T, String> {
}
