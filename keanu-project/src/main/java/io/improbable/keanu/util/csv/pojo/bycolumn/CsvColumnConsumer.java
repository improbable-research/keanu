package io.improbable.keanu.util.csv.pojo.bycolumn;

import java.util.List;
import java.util.function.BiConsumer;

/**
 * A CsvColumnConsumer is a function that takes some object
 * and some list of String values (a column) and applies the list to the
 * object. For example, if a given target object class type has
 * a public field named "something" that mapped to a csv column
 * header named "something" then this CsvColumnConsumer would set
 * the public field of an instance of (T) to the list of values also
 * provided.
 *
 * @param <T> target object type
 */
public interface CsvColumnConsumer<T> extends BiConsumer<T, List<String>> {
}
