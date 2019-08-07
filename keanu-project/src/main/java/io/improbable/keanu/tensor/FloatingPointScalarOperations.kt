package io.improbable.keanu.tensor

interface FloatingPointScalarOperations<T : Number> : NumberScalarOperations<T> {

    fun safeLogTimes(left: T, right: T): T

    fun logAddExp(left: T, right: T): T

    fun logAddExp2(left: T, right: T): T

}