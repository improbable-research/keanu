package io.improbable.keanu.tensor

interface FixedPointScalarOperations<T : Number> : NumberScalarOperations<T> {

    fun mod(left: T, right: T): T
}