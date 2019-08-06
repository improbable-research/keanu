package io.improbable.keanu.tensor.lng

import io.improbable.keanu.tensor.NumberScalarOperations

object LongScalarOperations : NumberScalarOperations<Long> {

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

}