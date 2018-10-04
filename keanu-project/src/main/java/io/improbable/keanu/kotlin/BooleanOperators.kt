package io.improbable.keanu.kotlin

/**
 * Custom operators for booleans.
 *
 * Built-in operators (&&, ||) cannot be overridden so infix functions are used instead.
 */
interface BooleanOperators<T> {

    infix fun and(that: T): T

    infix fun and(that: Boolean): T

    infix fun or(that: T): T

    infix fun or(that: Boolean): T

    fun not(): T

}

infix fun <T : BooleanOperators<T>> Boolean.and(that: T) = that.and(this)

infix fun <T : BooleanOperators<T>> Boolean.or(that: T) = that.or(this)

fun <T : BooleanOperators<T>> not(that: T) = that.not()
