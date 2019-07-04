# Keanu Tensors

The goal is for the Keanu tensor classes to be as feature complete as numpy while maintaining type safety.
For that to happen there are still several functions on the existing tensor classes that need
to be implemented and several new typed tensor classes yet to be implemented.

All tensor functions should as closely as possible mimic the numpy functions. This applies to
the broadcasting semantics as well.

The following functions are on the todo list:

## Tensor

### Functions to add

- stack(arrays[, axis, out])	Join a sequence of arrays along a new axis.
- column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
- dstack(tup)	Stack arrays in sequence depth wise (along third axis).
- hstack(tup)	Stack arrays in sequence horizontally (column wise).
- vstack(tup)	Stack arrays in sequence vertically (row wise).
- block(arrays)	Assemble an nd-array from nested lists of blocks.

- split(ary, indices_or_sections[, axis])	Split an array into multiple sub-arrays.
- array_split(ary, indices_or_sections[, axis])	Split an array into multiple sub-arrays.
- dsplit(ary, indices_or_sections)	Split array into multiple sub-arrays along the 3rd axis (depth).
- hsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays horizontally (column-wise).
- vsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays vertically (row-wise).

- tile(A, reps)	Construct an array by repeating A the number of times given by reps.
- repeat(a, repeats[, axis])	Repeat elements of an array.

- flip(m[, axis])	Reverse the order of elements in an array along the given axis.
- fliplr(m)	Flip array in the left/right direction.
- flipud(m)	Flip array in the up/down direction.
- roll(a, shift[, axis])	Roll array elements along a given axis.
- rot90(m[, k, axes])	Rotate an array by 90 degrees in the plane specified by axes.

- where(condition, [x, y])	Return elements chosen from x or y depending on condition.

- indices(dimensions[, dtype])	Return an array representing the indices of a grid.
- diag_indices(n[, ndim])	Return the indices to access the main diagonal of an array.
- diag_indices_from(arr)	Return the indices to access the main diagonal of an n-dimensional array.
- mask_indices(n, mask_func[, k])	Return the indices to access (n, n) arrays, given a masking function.
- tril_indices(n[, k, m])	Return the indices for the lower-triangle of an (n, m) array.
- tril_indices_from(arr[, k])	Return the indices for the lower-triangle of arr.
- triu_indices(n[, k, m])	Return the indices for the upper-triangle of an (n, m) array.
- triu_indices_from(arr[, k])	Return the indices for the upper-triangle of arr.

- take(a, indices[, axis, out, mode])	Take elements from an array along an axis.
- take_along_axis(arr, indices, axis)	Take values from the input array by matching 1d index and data slices.
- choose(a, choices[, out, mode])	Construct an array from an index array and a set of arrays to choose from.
- compress(condition, a[, axis, out])	Return selected slices of an array along given axis.
- diag(v[, k])	Extract a diagonal or construct a diagonal array.
- diagonal(a[, offset, axis1, axis2])	Return specified diagonals.
- select(condlist, choicelist[, default])	Return an array drawn from elements in choicelist, depending on conditions

- place(arr, mask, vals)	Change elements of an array based on conditional and input values.
- put(a, ind, v[, mode])	Replaces specified elements of an array with given values.
- put_along_axis(arr, indices, values, axis)	Put values into the destination array by matching 1d index and data slices.
- putmask(a, mask, values)	Changes elements of an array based on conditional and input values.
- fill_diagonal(a, val[, wrap])	Fill the main diagonal of the given array of any dimensionality.

#### Sorting

- sort(a[, axis, kind, order])	Return a sorted copy of an array.
- lexsort(keys[, axis])	Perform an indirect stable sort using a sequence of keys.
- argsort(a[, axis, kind, order])	Returns the indices that would sort an array.
- ndarray.sort([axis, kind, order])	Sort an array, in-place.
- msort(a)	Return a copy of an array sorted along the first axis.
- sort_complex(a)	Sort a complex array using the real part first, then the imaginary part.
- partition(a, kth[, axis, kind, order])	Return a partitioned copy of an array.
- argpartition(a, kth[, axis, kind, order])	Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.

#### Searching

- argwhere(a)	Find the indices of array elements that are non-zero, grouped by element.
- searchsorted(a, v[, side, sorter])	Find indices where elements should be inserted to maintain order.
- extract(condition, arr)	Return the elements of an array that satisfy some condition.

#### Counting

- count_nonzero(a[, axis])	Counts the number of non-zero values in the array a.

#### Static construction

- diag(v[, k])	Extract a diagonal or construct a diagonal array.
- diagflat(v[, k])	Create a two-dimensional array with the flattened input as a diagonal.
- tri(N[, M, k, dtype])	An array with ones at and below the given diagonal and zeros elsewhere.
- tril(m[, k])	Lower triangle of an array.
- triu(m[, k])	Upper triangle of an array.

## NumberTensor

- nonzero(a)	Return the indices of the elements that are non-zero.

### Functions to add

#### Arithmetic
- positive(x, /[, out, where, casting, order, …])	Numerical positive, element-wise.
- negative(x, /[, out, where, casting, order, …])	Numerical negative, element-wise.
- true_divide(x1, x2, /[, out, where, …])	Returns a true division of the inputs, element-wise.
- floor_divide(x1, x2, /[, out, where, …])	Return the largest integer smaller or equal to the division of the inputs.
- float_power(x1, x2, /[, out, where, …])	First array elements raised to powers from second array, element-wise.
- fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
- mod(x1, x2, /[, out, where, casting, order, …])	Return element-wise remainder of division.
- modf(x[, out1, out2], / [[, out, where, …])	Return the fractional and integral parts of an array, element-wise.
- remainder(x1, x2, /[, out, where, casting, …])	Return element-wise remainder of division.
- divmod(x1, x2[, out1, out2], / [[, out, …])	Return element-wise quotient and remainder simultaneously.

#### Other

- convolve(a, v[, mode])	Returns the discrete, linear convolution of two one-dimensional sequences.
- clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
- cbrt(x, /[, out, where, casting, order, …])	Return the cube-root of an array, element-wise.
- square(x, /[, out, where, casting, order, …])	Return the element-wise square of the input.
- fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
- sign(x, /[, out, where, casting, order, …])	Returns an element-wise indication of the sign of a number.
- heaviside(x1, x2, /[, out, where, casting, …])	Compute the Heaviside step function.
- fmax(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
- fmin(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.
- nan_to_num(x[, copy])	Replace NaN with zero and infinity with large finite numbers.
- real_if_close(a[, tol])	If complex input returns a real array if complex parts are close to zero.
- interp(x, xp, fp[, left, right, period])	One-dimensional linear interpolation.

- amin(a[, axis, out, keepdims, initial])	Return the minimum of an array or minimum along an axis.
- amax(a[, axis, out, keepdims, initial])	Return the maximum of an array or maximum along an axis.
- nanmin(a[, axis, out, keepdims])	Return minimum of an array or minimum along an axis, ignoring any NaNs.
- nanmax(a[, axis, out, keepdims])	Return the maximum of an array or maximum along an axis, ignoring any NaNs.
- ptp(a[, axis, out, keepdims])	Range of values (maximum - minimum) along an axis.
- percentile(a, q[, axis, out, …])	Compute the q-th percentile of the data along the specified axis.
- nanpercentile(a, q[, axis, out, …])	Compute the qth percentile of the data along the specified axis, while ignoring nan values.
- quantile(a, q[, axis, out, overwrite_input, …])	Compute the q-th quantile of the data along the specified axis.
- nanquantile(a, q[, axis, out, …])	Compute the qth quantile of the data along the specified axis, while ignoring nan values.

- median(a[, axis, out, overwrite_input, keepdims])	Compute the median along the specified axis.
- average(a[, axis, weights, returned])	Compute the weighted average along the specified axis.
- mean(a[, axis, dtype, out, keepdims])	Compute the arithmetic mean along the specified axis.
- std(a[, axis, dtype, out, ddof, keepdims])	Compute the standard deviation along the specified axis.
- var(a[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis.
- nanmedian(a[, axis, out, overwrite_input, …])	Compute the median along the specified axis, while ignoring NaNs.
- nanmean(a[, axis, dtype, out, keepdims])	Compute the arithmetic mean along the specified axis, ignoring NaNs.
- nanstd(a[, axis, dtype, out, ddof, keepdims])	Compute the standard deviation along the specified axis, while ignoring NaNs.
- nanvar(a[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis, while ignoring NaNs.

- corrcoef(x[, y, rowvar, bias, ddof])	Return Pearson product-moment correlation coefficients.
- correlate(a, v[, mode])	Cross-correlation of two 1-dimensional sequences.
- cov(m[, y, rowvar, bias, ddof, fweights, …])	Estimate a covariance matrix, given data and weights.

- histogram(a[, bins, range, normed, weights, …])	Compute the histogram of a set of data.
- histogram2d(x, y[, bins, range, normed, …])	Compute the bi-dimensional histogram of two data samples.
- histogramdd(sample[, bins, range, normed, …])	Compute the multidimensional histogram of some data.
- bincount(x[, weights, minlength])	Count number of occurrences of each value in array of non-negative ints.
- histogram_bin_edges(a[, bins, range, weights])	Function to calculate only the edges of the bins used by the histogram function.
- digitize(x, bins[, right])	Return the indices of the bins to which each value in input array belongs.

## FloatingPointTensor

Missing types:
- FloatTensor

### Functions to add

#### Linear Alg

- dot(a, b[, out])	Dot product of two arrays.
- linalg.multi_dot(arrays)	Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.
- vdot(a, b)	Return the dot product of two vectors.
- inner(a, b)	Inner product of two arrays.
- outer(a, b[, out])	Compute the outer product of two vectors.
- einsum(subscripts, *operands[, out, dtype, …])	Evaluates the Einstein summation convention on the operands.
- einsum_path(subscripts, *operands[, optimize])	Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.
- linalg.matrix_power(a, n)	Raise a square matrix to the (integer) power n.
- kron(a, b)	Kronecker product of two arrays.

- linalg.qr(a[, mode])	Compute the qr factorization of a matrix.
- linalg.svd(a[, full_matrices, compute_uv])	Singular Value Decomposition.

- linalg.eig(a)	Compute the eigenvalues and right eigenvectors of a square array.
- linalg.eigh(a[, UPLO])	Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
- linalg.eigvals(a)	Compute the eigenvalues of a general matrix.
- linalg.eigvalsh(a[, UPLO])	Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

- linalg.norm(x[, ord, axis, keepdims])	Matrix or vector norm.
- linalg.cond(x[, p])	Compute the condition number of a matrix.
- linalg.det(a)	Compute the determinant of an array.
- linalg.matrix_rank(M[, tol, hermitian])	Return matrix rank of array using SVD method
- linalg.slogdet(a)	Compute the sign and (natural) logarithm of the determinant of an array.
- trace(a[, offset, axis1, axis2, dtype, out])	Return the sum along diagonals of the array.

- linalg.solve(a, b)	Solve a linear matrix equation, or system of linear scalar equations.
- linalg.tensorsolve(a, b[, axes])	Solve the tensor equation a x = b for x.
- linalg.lstsq(a, b[, rcond])	Return the least-squares solution to a linear matrix equation.
- linalg.inv(a)	Compute the (multiplicative) inverse of a matrix.
- linalg.pinv(a[, rcond])	Compute the (Moore-Penrose) pseudo-inverse of a matrix.
- linalg.tensorinv(a[, ind])	Compute the ‘inverse’ of an N-dimensional array.

#### FFT lib
- Lots of FFT related stuff. Maybe add?

#### Comparison

- isfinite(x, /[, out, where, casting, order, …])	Test element-wise for finiteness (not infinity or not Not a Number).
- isinf(x, /[, out, where, casting, order, …])	Test element-wise for positive or negative infinity.
- isnan(x, /[, out, where, casting, order, …])	Test element-wise for NaN and return result as a boolean array.
- isnat(x, /[, out, where, casting, order, …])	Test element-wise for NaT (not a time) and return result as a boolean array.
- isneginf(x[, out])	Test element-wise for negative infinity, return result as bool array.
- isposinf(x[, out])	Test element-wise for positive infinity, return result as bool array.

#### Trig

- hypot(x1, x2, /[, out, where, casting, …])	Given the “legs” of a right triangle, return its hypotenuse.
- degrees(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
- radians(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
- unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.
- deg2rad(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
- rad2deg(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
- sinc(x)	Return the sinc function.

#### Rounding

- around(a[, decimals, out])	Evenly round to the given number of decimals.
- round_(a[, decimals, out])	Round an array to the given number of decimals.
- rint(x, /[, out, where, casting, order, …])	Round elements of the array to the nearest integer.
- fix(x[, out])	Round to nearest integer towards zero.
- trunc(x, /[, out, where, casting, order, …])	Return the truncated value of the input, element-wise.

#### Sum, Product, Differences

- nanprod(a[, axis, dtype, out, keepdims])	Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
- nansum(a[, axis, dtype, out, keepdims])	Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
- nancumprod(a[, axis, dtype, out])	Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.
- nancumsum(a[, axis, dtype, out])	Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
- diff(a[, n, axis, prepend, append])	Calculate the n-th discrete difference along the given axis.
- ediff1d(ary[, to_end, to_begin])	The differences between consecutive elements of an array.
- gradient(f, *varargs, **kwargs)	Return the gradient of an N-dimensional array.
- cross(a, b[, axisa, axisb, axisc, axis])	Return the cross product of two (arrays of) vectors.
- trapz(y[, x, dx, axis])	Integrate along the given axis using the composite trapezoidal rule.

#### Arithmetic

#### Windowing

- bartlett(M)	Return the Bartlett window.
- blackman(M)	Return the Blackman window.
- hamming(M)	Return the Hamming window.
- hanning(M)	Return the Hanning window.
- kaiser(M, beta)	Return the Kaiser window.

### Static construction

- logspace(start, stop[, num, endpoint, base, …])	Return numbers spaced evenly on a log scale.
- geomspace(start, stop[, num, endpoint, …])	Return numbers spaced evenly on a log scale (a geometric progression).
- meshgrid(*xi, **kwargs)	Return coordinate matrices from coordinate vectors.
- vander(x[, N, increasing])	Generate a Vandermonde matrix.

### DoubleTensor

#### Functions to add

- All of the floating point additions

## FixedPointTensor

Missing types:
- ByteTensor
- ShortTensor
- CharTensor
- LongTensor

### Functions to add

- bitwise_and(x1, x2, /[, out, where, …])	Compute the bit-wise AND of two arrays element-wise.
- bitwise_or(x1, x2, /[, out, where, casting, …])	Compute the bit-wise OR of two arrays element-wise.
- bitwise_xor(x1, x2, /[, out, where, …])	Compute the bit-wise XOR of two arrays element-wise.
- invert(x, /[, out, where, casting, order, …])	Compute bit-wise inversion, or bit-wise NOT, element-wise.
- left_shift(x1, x2, /[, out, where, casting, …])	Shift the bits of an integer to the left.
- right_shift(x1, x2, /[, out, where, …])	Shift the bits of an integer to the right.

- packbits(myarray[, axis])	Packs the elements of a binary-valued array into bits in a uint8 array.
- unpackbits(myarray[, axis])	Unpacks elements of a uint8 array into a binary-valued output array.

- lcm(x1, x2, /[, out, where, casting, order, …])	Returns the lowest common multiple of |x1| and |x2|
- gcd(x1, x2, /[, out, where, casting, order, …])	Returns the greatest common divisor of |x1| and |x2|

### IntegerTensor

#### Functions to add

- all of the fixed point functions

## BooleanTensor

### Functions to add


