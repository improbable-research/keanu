package io.improbable.keanu.tensor.lng

import io.improbable.keanu.tensor.FixedPointScalarOperations

object LongScalarOperations : FixedPointScalarOperations<Long> {

    override fun mod(left: Long, right: Long): Long {
        return left % right
    }

    // Number Ops
    override fun sub(left: Long, right: Long): Long {
        return left - right
    }

    override fun add(left: Long, right: Long): Long {
        return left + right
    }

    override fun rsub(left: Long, right: Long): Long {
        return right - left
    }

    override fun mul(left: Long, right: Long): Long {
        return left * right
    }

    override fun div(left: Long, right: Long): Long {
        return left / right
    }

    override fun rdiv(left: Long, right: Long): Long {
        return right / left
    }

    override fun pow(left: Long, right: Long): Long {
        if (right == 0L) {
            return 1L
        }

        var result = left
        for (i in 0 until right - 1) {
            result *= left
        }

        return result
    }

    override fun abs(value: Long): Long {
        return kotlin.math.abs(value)
    }

    override fun sign(value: Long): Long {
        return if (value > 0L) 1L else if (value < 0L) -1L else 0L
    }

    override fun unaryMinus(value: Long): Long {
        return -value
    }

    //Comparisons

    override fun equalToMask(left: Long, right: Long): Long {
        return if (left == right) 1 else 0
    }

    override fun gtMask(left: Long, right: Long): Long {
        return if (left > right) 1 else 0
    }

    override fun gteMask(left: Long, right: Long): Long {
        return if (left >= right) 1 else 0
    }

    override fun ltMask(left: Long, right: Long): Long {
        return if (left < right) 1 else 0
    }

    override fun lteMask(left: Long, right: Long): Long {
        return if (left <= right) 1 else 0
    }

    override fun equalToWithinEpsilon(left: Long, right: Long, epsilon: Long): Boolean {
        return kotlin.math.abs(left - right) <= epsilon
    }

    override fun equalTo(left: Long, right: Long): Boolean {
        return left == right
    }

    override fun gt(left: Long, right: Long): Boolean {
        return left > right
    }

    override fun gte(left: Long, right: Long): Boolean {
        return left >= right
    }

    override fun lt(left: Long, right: Long): Boolean {
        return left < right
    }

    override fun lte(left: Long, right: Long): Boolean {
        return left <= right
    }

    override fun max(left: Long, right: Long): Long {
        return kotlin.math.max(left, right)
    }

    override fun min(left: Long, right: Long): Long {
        return kotlin.math.min(left, right)
    }

}