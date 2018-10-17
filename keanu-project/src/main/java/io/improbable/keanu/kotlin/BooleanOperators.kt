package io.improbable.keanu.kotlin

/**
 * Custom operators for booleans.
 *
 * Built-in binary operators (&&, ||) cannot be overridden so infix functions are used instead. [See this
 * discussion for more info](https://discuss.kotlinlang.org/t/overloading-and-and-or-operators/1623/2).
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
