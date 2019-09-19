package io.improbable.keanu.tensor.intgr

import io.improbable.keanu.tensor.FixedPointScalarOperations

object IntegerScalarOperations : FixedPointScalarOperations<Int> {

    override fun mod(left: Int, right: Int): Int {
        return left % right
    }

    // Number Ops
    override fun sub(left: Int, right: Int): Int {
        return left - right
    }

    override fun add(left: Int, right: Int): Int {
        return left + right
    }

    override fun rsub(left: Int, right: Int): Int {
        return right - left
    }

    override fun mul(left: Int, right: Int): Int {
        return left * right
    }

    override fun div(left: Int, right: Int): Int {
        return left / right
    }

    override fun rdiv(left: Int, right: Int): Int {
        return right / left
    }

    override fun pow(left: Int, right: Int): Int {
        if (right == 0) {
            return 1
        }

        var result = left
        for (i in 0 until right - 1) {
            result *= left
        }

        return result
    }

    override fun abs(value: Int): Int {
        return kotlin.math.abs(value)
    }

    override fun sign(value: Int): Int {
        return if (value > 0) 1 else if (value < 0) -1 else 0
    }

    override fun unaryMinus(value: Int): Int {
        return -value
    }

    //Comparisons

    override fun equalToMask(left: Int, right: Int): Int {
        return if (left == right) 1 else 0
    }

    override fun gtMask(left: Int, right: Int): Int {
        return if (left > right) 1 else 0
    }

    override fun gteMask(left: Int, right: Int): Int {
        return if (left >= right) 1 else 0
    }

    override fun ltMask(left: Int, right: Int): Int {
        return if (left < right) 1 else 0
    }

    override fun lteMask(left: Int, right: Int): Int {
        return if (left <= right) 1 else 0
    }

    override fun equalToWithinEpsilon(left: Int, right: Int, epsilon: Int): Boolean {
        return kotlin.math.abs(left - right) <= epsilon
    }

    override fun equalTo(left: Int, right: Int): Boolean {
        return left == right
    }

    override fun gt(left: Int, right: Int): Boolean {
        return left > right
    }

    override fun gte(left: Int, right: Int): Boolean {
        return left >= right
    }

    override fun lt(left: Int, right: Int): Boolean {
        return left < right
    }

    override fun lte(left: Int, right: Int): Boolean {
        return left <= right
    }

    override fun max(left: Int, right: Int): Int {
        return kotlin.math.max(left, right)
    }

    override fun min(left: Int, right: Int): Int {
        return kotlin.math.min(left, right)
    }

}