package utils

import (
	"fmt"
)

/*
as is
- both a and b are 1-d array, it is inner product of vetors
to do list
- use generic
- b scalar, equivalent to multiply
- a is n-d and b is 1-d array, it is a sum product over a rows and b
- n-d * m-d or 1-d * n-d, matrix multiplication
*/
func Dot(a, b []float64) (float64, error) {
	alen := len(a)
	blen := len(b)
	if alen != blen {
		return .0, fmt.Errorf("Dot: not match dimension. (a=%d(col) b=%d(col))", alen, blen)
	}

	sum := 0.0
	for i, ac := range a {
		sum += ac * b[i]
	}

	return sum, nil
}
