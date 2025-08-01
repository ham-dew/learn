package utils

import (
	"fmt"
	"math"
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

func Transpose(source [][]float64) [][]float64 {
	c := len(source[0])

	res := make([][]float64, len(source))
	for i := range res {
		res[i] = make([]float64, c)
	}

	for i, row := range source {
		for j, col := range row {
			res[j][i] = col
		}
	}

	return res
}

func EuclideanDistance(p1, p2 []float64) (float64, error) {
	if len(p1) != len(p2) {
		return -1, fmt.Errorf("mismatched dimensions")
	}

	total := 0.0

	for i, x := range p1 {
		total += math.Pow(x-p2[i], 2)
	}

	return math.Sqrt(total), nil
}

func MaxIndex(arr []float64) int {
	idx := 0
	max := math.Inf(-1)

	for i, v := range arr {
		if v > max {
			max = v
			idx = i
		}
	}

	return idx
}

func MinIndex(arr []float64) int {
	idx := 0
	min := math.MaxFloat64

	for i, v := range arr {
		if v < min {
			min = v
			idx = i
		}
	}

	return idx
}
