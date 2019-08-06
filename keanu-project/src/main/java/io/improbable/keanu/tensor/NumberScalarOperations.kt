package io.improbable.keanu.tensor

interface NumberScalarOperations<T : Number> {

    //Number ops

    fun add(left: T, right: T): T
    fun sub(left: T, right: T): T
    fun rsub(left: T, right: T): T
    fun mul(left: T, right: T): T
    fun div(left: T, right: T): T
    fun rdiv(left: T, right: T): T
    fun pow(left: T, right: T): T

    //Comparisons

    fun equalTo(left: T, right: T): Boolean
    fun gt(left: T, right: T): Boolean
    fun gte(left: T, right: T): Boolean
    fun lt(left: T, right: T): Boolean
    fun lte(left: T, right: T): Boolean

    fun equalToMask(left: T, right: T): T
    fun gtMask(left: T, right: T): T
    fun gteMask(left: T, right: T): T
    fun ltMask(left: T, right: T): T
    fun lteMask(left: T, right: T): T
}