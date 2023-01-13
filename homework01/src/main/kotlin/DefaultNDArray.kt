import java.lang.Integer.min

interface NDArray : SizeAware, DimentionAware {
    /*
     * Получаем значение по индексу point
     *
     * Если размерность point не равна размерности NDArray
     * бросаем IllegalPointDimensionException
     *
     * Если позиция по любой из размерностей некорректна с точки зрения
     * размерности NDArray, бросаем IllegalPointCoordinateException
     */
    fun at(point: Point): Int

    /*
     * Устанавливаем значение по индексу point
     *
     * Если размерность point не равна размерности NDArray
     * бросаем IllegalPointDimensionException
     *
     * Если позиция по любой из размерностей некорректна с точки зрения
     * размерности NDArray, бросаем IllegalPointCoordinateException
     */
    fun set(point: Point, value: Int)

    /*
     * Копируем текущий NDArray
     *
     */
    fun copy(): NDArray

    /*
     * Создаем view для текущего NDArray
     *
     * Ожидается, что будет создан новая реализация  интерфейса.
     * Но она не должна быть видна в коде, использующем эту библиотеку как внешний артефакт
     *
     * Должна быть возможность делать view над view.
     *
     * In-place-изменения над view любого порядка видна в оригнале и во всех view
     *
     * Проблемы thread-safety игнорируем
     */
    fun view(): NDArray

    /*
     * In-place сложение
     *
     * Размерность other либо идентична текущей, либо на 1 меньше
     * Если она на 1 меньше, то по всем позициям, кроме "лишней", она должна совпадать
     *
     * Если размерности совпадают, то делаем поэлементное сложение
     *
     * Если размерность other на 1 меньше, то для каждой позиции последней размерности мы
     * делаем поэлементное сложение
     *
     * Например, если размерность this - (10, 3), а размерность other - (10), то мы для три раза прибавим
     * other к каждому срезу последней размерности
     *
     * Аналогично, если размерность this - (10, 3, 5), а размерность other - (10, 5), то мы для пять раз прибавим
     * other к каждому срезу последней размерности
     */
    fun add(other: NDArray)

    /*
     * Умножение матриц. Immutable-операция. Возвращаем NDArray
     *
     * Требования к размерности - как для умножения матриц.
     *
     * this - обязательно двумерна
     *
     * other - может быть двумерной, с подходящей размерностью, равной 1 или просто вектором
     *
     * Возвращаем новую матрицу (NDArray размерности 2)
     *
     */
    fun dot(other: NDArray): NDArray
}

/*
 * Базовая реализация NDArray
 *
 * Конструкторы должны быть недоступны клиенту
 *
 * Инициализация - через factory-методы ones(shape: Shape), zeros(shape: Shape) и метод copy
 */
class DefaultNDArray private constructor(val data: IntArray, private val shape: Shape) : NDArray {
    companion object {
        fun zeros(shape: Shape): NDArray {
            return DefaultNDArray(IntArray(shape.size), shape)
        }

        fun ones(shape: Shape): NDArray {
            return DefaultNDArray(IntArray(shape.size) { 1 }, shape)
        }
    }

    override val ndim: Int
        get() = shape.ndim

    override fun dim(i: Int): Int {
        return shape.dim(i)
    }

    override val size: Int
        get() = data.size

    private fun getIndex(point: Point): Int {
        if (point.ndim != ndim) {
            throw NDArrayException.IllegalPointDimensionException(point.ndim, ndim)
        }
        var index = 0
        var currBlockSize: Int = size
        for (i in 0 until point.ndim) {
            if (point.dim(i) < 0 || point.dim(i) >= shape.dim(i)) {
                throw NDArrayException.IllegalPointCoordinateException(point.dim(i), shape.dim(i));
            }
            currBlockSize /= shape.dim(i)
            index += point.dim(i) * currBlockSize
        }
        return index;
    }

    override fun at(point: Point): Int {
        return data[getIndex(point)];
    }

    override fun set(point: Point, value: Int) {
        data[getIndex(point)] = value
    }

    override fun copy(): NDArray {
        return DefaultNDArray(data.copyOf(), shape)
    }

    override fun view(): NDArray {
        return ViewNDArray(this)
    }

    private fun addArray(other: NDArray, startIndex: Int, otherDims: IntArray, blockSize: Int, kdim: Int) {
        if (kdim < ndim) {
            val newBlockSize = blockSize / dim(kdim)
            val minDim = if (kdim < other.ndim) min(dim(kdim), other.dim(kdim)) else dim(kdim)
            for (i in 0 until minDim) {
                val newStartIndex = startIndex + i * newBlockSize
                if (kdim < other.ndim) otherDims[kdim] = i
                addArray(other, newStartIndex, otherDims.copyOf(), newBlockSize, kdim + 1)
            }
        } else {
            data[startIndex] += other.at(DefaultPoint(*otherDims))
        }
    }

    override fun add(other: NDArray) {
        if (ndim == other.ndim || ndim == other.ndim + 1) {
            addArray(other, 0, IntArray(other.ndim), size, 0);
        } else {
            throw NDArrayException.IllegalNDArrayDimensionException(ndim, other.ndim, other.ndim + 1);
        }
    }

    override fun dot(other: NDArray): NDArray {
        if (ndim != 2) {
            throw NDArrayException.IllegalNDArrayDimensionException(ndim, 2)
        }
        if (other.ndim > 2) {
            throw NDArrayException.IllegalNDArrayDimensionException(other.ndim, 2)
        }
        if (dim(1) != other.dim(0)) {
            throw NDArrayException.IllegalPointDimensionException(dim(1), other.dim(0))
        }

        val res: NDArray
        if (other.ndim > 1) {
            res = zeros(DefaultShape(dim(0), other.dim(1)))
            for (i in 0 until dim(0)) {
                for (j in 0 until other.dim(1)) {
                    var value = 0
                    for (k in 0 until dim(1)) {
                        value += this.at(DefaultPoint(i, k)) * other.at(DefaultPoint(k, j))
                    }
                    res.set(DefaultPoint(i, j), value)
                }
            }
        } else {
            res = zeros(DefaultShape(dim(0)))
            for (i in 0 until dim(0)) {
                var value = 0
                for (k in 0 until dim(1)) {
                    value += this.at(DefaultPoint(i, k)) * other.at(DefaultPoint(k))
                }
                res.set(DefaultPoint(i), value)
            }
        }
        return res
    }

    override fun toString(): String {
        return "DefaultNDArray(data=${data.contentToString()})"
    }

}

internal class ViewNDArray(val source: NDArray) : NDArray {

    override val size: Int
        get() = source.size
    override val ndim: Int
        get() = source.ndim

    override fun dim(i: Int): Int {
        return source.dim(i)
    }

    override fun at(point: Point): Int {
        return source.at(point)
    }

    override fun set(point: Point, value: Int) {
        source.set(point, value)
    }

    override fun copy(): NDArray {
        return source.copy();
    }

    override fun view(): NDArray {
        return source.view()
    }

    override fun add(other: NDArray) {
        source.add(other)
    }

    override fun dot(other: NDArray): NDArray {
        return source.dot(other)
    }

}

sealed class NDArrayException(reason: String = "") : Exception(reason) {
    /* TODO: реализовать требуемые исключения */
    class IllegalPointCoordinateException(current: Int, max: Int) :
        NDArrayException("Illegal point dimension exception: current value: $current, expected value from range 0 to $max")

    class IllegalPointDimensionException(current: Int, expected: Int) :
        NDArrayException("Illegal point dimension exception: current value = $current, expected = $expected")

    class IllegalNDArrayDimensionException(current: Int, vararg expected: Int) :
        NDArrayException("Illegal NDArray dimension exception: current value = $current, expected value from ${expected.contentToString()}")
}