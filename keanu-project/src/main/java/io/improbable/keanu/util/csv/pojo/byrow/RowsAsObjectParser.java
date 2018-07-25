package io.improbable.keanu.util.csv.pojo.byrow;

import io.improbable.keanu.util.csv.pojo.PublicFieldMatcher;
import io.improbable.keanu.util.csv.pojo.SetterMatcher;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;


/**
 * This class parses csv lines to a specified plain old java object (POJO). This
 * can be done to a POJO that has a public field OR a setter method OR
 * a an annotated field/method.
 */
public class RowsAsObjectParser<T> {

    private static final boolean IGNORE_MISSING_FIELDS_DEFAULT = false;

    private final Class<T> base;
    private final Stream<List<String>> inputStream;
    private final List<String> csvTitles;

    public RowsAsObjectParser(Class<T> base, Stream<List<String>> inputStream, List<String> csvTitles) {
        this.base = base;
        this.inputStream = inputStream;
        this.csvTitles = csvTitles;
    }

    public List<T> load(boolean ignoreUnmatchedFields) {
        return stream(ignoreUnmatchedFields)
            .collect(toList());
    }

    public List<T> load() {
        return load(IGNORE_MISSING_FIELDS_DEFAULT);
    }

    public Stream<T> stream(boolean ignoreUnmatchedFields) {
        return stream(base, inputStream, csvTitles, ignoreUnmatchedFields);
    }

    public Stream<T> stream() {
        return stream(IGNORE_MISSING_FIELDS_DEFAULT);
    }

    public static <T> Stream<T> stream(Class<T> base,
                                       Stream<List<String>> inputStream,
                                       List<String> csvTitles) {

        return stream(base, inputStream, csvTitles, IGNORE_MISSING_FIELDS_DEFAULT);
    }

    public static <T> Stream<T> stream(Class<T> base,
                                       Stream<List<String>> csvLinesAsTokens,
                                       List<String> csvTitles,
                                       boolean ignoreUnmatchedFields) {

        List<CsvCellConsumer<T>> columnConsumers = getCellConsumers(base, csvTitles, ignoreUnmatchedFields);

        return csvLinesAsTokens
            .map(csvTokens -> deserialize(csvTokens, columnConsumers, base));
    }

    private static <T> List<CsvCellConsumer<T>> getCellConsumers(Class<T> base,
                                                                 List<String> fieldTitles,
                                                                 boolean ignoreUnmatchedFields) {

        List<CsvCellConsumer<T>> columnConsumers = new ArrayList<>();
        CsvCellConsumer<T> defaultConsumer = (target, value) -> {
        };

        List<Field> potentialFields = new ArrayList<>(Arrays.asList(base.getFields()));
        List<Method> potentialSetters = new ArrayList<>(Arrays.asList(base.getMethods()));

        for (String title : fieldTitles) {

            Optional<CsvCellConsumer<T>> consumerForTitle;

            consumerForTitle = PublicFieldMatcher.getFieldCellConsumer(title, potentialFields);

            if (!consumerForTitle.isPresent()) {
                consumerForTitle = SetterMatcher.getSetterCellConsumer(title, potentialSetters);
            }

            if (!consumerForTitle.isPresent() && !ignoreUnmatchedFields) {
                throw new IllegalArgumentException("Unable to find field for csv data \"" + title + "\"");
            }

            columnConsumers.add(consumerForTitle.orElse(defaultConsumer));
        }
        return columnConsumers;
    }

    private static <T> T deserialize(List<String> csvTokens,
                                     List<CsvCellConsumer<T>> fieldMappers,
                                     Class<T> base) {

        try {
            T target = base.newInstance();

            for (int i = 0; i < csvTokens.size(); i++) {
                BiConsumer<T, String> fieldConsumer = fieldMappers.get(i);
                if (fieldConsumer != null) {
                    fieldConsumer.accept(target, csvTokens.get(i));
                }
            }

            return target;
        } catch (IllegalAccessException | InstantiationException e) {
            throw new IllegalStateException(e);
        }
    }

}
