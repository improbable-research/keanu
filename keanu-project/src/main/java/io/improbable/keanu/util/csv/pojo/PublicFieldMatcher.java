package io.improbable.keanu.util.csv.pojo;

import io.improbable.keanu.util.csv.pojo.bycolumn.CsvColumnConsumer;
import io.improbable.keanu.util.csv.pojo.byrow.CsvCellConsumer;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;

/**
 * This finds an appropriate POJO public field for a given csv title.
 * <p>
 * Example:
 * <p>
 * id,name
 * 0,abc
 * 1,efg
 * <p>
 * and a POJO
 * <p>
 * public class SomePOJO {
 * public String name;
 * public int id;
 * ...
 */
public class PublicFieldMatcher {

    private PublicFieldMatcher() {
    }

    public static <T> Optional<CsvCellConsumer<T>> getFieldCellConsumer(String title, List<Field> potentialFields) {

        final Optional<Field> matchingField = findMatchingFieldName(title.trim(), potentialFields);

        return matchingField
            .map(PublicFieldMatcher::createCellConsumerForField);
    }

    private static <T> CsvCellConsumer<T> createCellConsumerForField(Field matchingField) {
        return (target, value) -> {

            Object convertedValue = CsvCellDeserializer.convertToAppropriateType(value, matchingField.getType());
            try {
                matchingField.set(target, convertedValue);
            } catch (IllegalAccessException e) {
                throw new IllegalArgumentException(e);
            }
        };
    }

    public static <T> Optional<CsvColumnConsumer<T>> getFieldColumnConsumer(String title, List<Field> potentialFields) {

        final Optional<Field> matchingField = findMatchingFieldName(title.trim(), potentialFields);

        return matchingField
            .map(PublicFieldMatcher::createColumnConsumerForField);
    }

    private static <T> CsvColumnConsumer<T> createColumnConsumerForField(Field matchingField) {
        return (target, value) -> {

            Object convertedValue = CsvColumnDeserializer.convertToAppropriateType(value, matchingField.getType());
            try {
                matchingField.set(target, convertedValue);
            } catch (IllegalAccessException e) {
                throw new IllegalArgumentException(e);
            }
        };
    }

    private static Optional<Field> findMatchingFieldName(String title, List<Field> potentials) {
        return potentials.stream()
            .filter(field -> isNameMatch(field, title) || hasCsvPropertyAnnotationWithName(field, title))
            .findFirst();
    }

    private static boolean isNameMatch(Field field, String title) {
        return field.getName().equalsIgnoreCase(title);
    }

    private static boolean hasCsvPropertyAnnotationWithName(Field field, String title) {

        if (field.isAnnotationPresent(CsvProperty.class)) {
            CsvProperty annotation = field.getAnnotation(CsvProperty.class);
            return annotation.value().equals(title);
        }

        return false;
    }
}
