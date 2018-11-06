package io.improbable.keanu.annotation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Annotation used for vertex classes to specify how they should be exported to a DOT file.
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface DisplayInformationForOutput {
    String displayName() default "";
    boolean displayHyperparameterInfo() default false;
}
