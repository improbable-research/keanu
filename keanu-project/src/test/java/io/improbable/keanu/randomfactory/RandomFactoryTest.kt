package io.improbable.keanu.randomfactory

import io.improbable.keanu.kotlin.ArithmeticDouble
import io.improbable.keanu.kotlin.DoubleOperators
import io.improbable.keanu.vertices.dbl.DoubleVertex
import junit.framework.TestCase
import org.junit.Test
import java.util.*

class RandomFactoryTest {

    @Test
    fun arithmeticAndVertexModelsAreEquivalent() {
        val modelA = getArithmeticDoubleTestModel()
        val modelB = getDoubleVertexTestModel()

        val resultA: ArithmeticDouble = modelA.addGaussians(1.0, 1.0, 1.0)
        val resultB: DoubleVertex = modelB.addGaussians(1.0, 1.0, 1.0)

        TestCase.assertEquals(resultA.value, resultB.value, 0.0)
    }

    @Test
    fun testDefault() {
        val modelA = getArithmeticDoubleTestModel()
        val modelB = getDoubleVertexTestModel()

        var sumA = 0.0
        var sumB = 0.0
        val num = 10000

        for (i: Int in 0..num) {
            sumA += modelA.addDefaultGaussians().value
            sumB += modelB.addDefaultGaussians().value
        }

        val meanA = sumA / num
        val meanB = sumB / num

        TestCase.assertEquals(meanA, meanB, 0.0)
    }

    private fun getArithmeticDoubleTestModel(): SimpleModel<ArithmeticDouble> {
        val arithmeticDoubleFactory = RandomDoubleFactory()
        arithmeticDoubleFactory.setRandom(Random(1))

        return SimpleModel<ArithmeticDouble>(arithmeticDoubleFactory)
    }

    private fun getDoubleVertexTestModel(): SimpleModel<DoubleVertex> {
        val doubleVertexFactory = DoubleVertexFactory()
        doubleVertexFactory.setRandom(Random(1))
        
        return SimpleModel<DoubleVertex>(doubleVertexFactory)
    }

    inner class SimpleModel<T : DoubleOperators<T>>(val randomFactory: RandomFactory<T>) {

        fun addGaussians(muA: Double, muB: Double, sigma: Double): T {
            val a = randomFactory.nextGaussian(muA, sigma)
            val b = randomFactory.nextGaussian(muB, sigma)
            return a + b
        }

        fun addDefaultGaussians(): T {
            val a = randomFactory.nextGaussian()
            val b = randomFactory.nextGaussian()
            return a + b
        }
    }
}
