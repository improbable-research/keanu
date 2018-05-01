package io.improbable.keanu.util.csv.pojo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;

import static io.improbable.keanu.util.csv.pojo.ColumnDeserializer.convertToAppropriateType;

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
class PublicFieldMatcher {

    private static final Logger log = LoggerFactory.getLogger(PublicFieldMatcher.class);

    private PublicFieldMatcher() {
    }

    static <T> Optional<CsvColumnConsumer<T>> getFieldConsumer(String title, List<Field> potentialFields) {

        final Optional<Field> matchingField = findMatchingFieldName(title.trim(), potentialFields);

        return matchingField
                .map(PublicFieldMatcher::createColumnConsumerForField);
    }

    private static <T> CsvColumnConsumer<T> createColumnConsumerForField(Field matchingField) {
        return (target, value) -> {

            Object convertedValue = convertToAppropriateType(value, matchingField.getType());
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
