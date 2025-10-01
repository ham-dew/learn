package utils

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// implementation numpy.dot
func Dot(a, b *mat.Dense) (res *mat.Dense, err error) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if br == 1 && bc == 1 { // if b is 0-d (scalar), it is equivalent to multiply
		scalar := b.At(0, 0)
		res = mat.NewDense(ar, ac, nil)
		res.Apply(func(_, _ int, v float64) float64 { return v * scalar }, a)
		return res, nil
	}
	if ar == 1 && br == 1 { // if both a and b are 1-d array, it is inner product of vetors
		if ac != bc {
			return nil, fmt.Errorf("Dot: not match 1d array sum column length. (a=%d b=%d)", ac, bc)
		}
		mul := new(mat.Dense)
		mul.MulElem(a, b)
		res = mat.NewDense(1, 1, []float64{mat.Sum(mul)})
		return res, nil
	}
	if br == 1 { // a is n-d and b is 1-d array, it is a sum product over a rows and b
		if ac != bc {
			return nil, fmt.Errorf("Dot: not match dimension. (a=%d(col) b=%d(col))", ac, bc)
		}
		inner := make([]float64, 0)
		for i := 0; i < ar; i++ {
			inner = append(inner, mat.Dot(a.RowView(i), b.RowView(0)))
		}
		res = mat.NewDense(br, ar, inner)
		return res, nil
	}
	// n-d * m-d or 1-d * n-d, matrix multiplication
	if ac != br {
		return nil, fmt.Errorf("Dot: not match dimension. (a=%d(col) b=%d(row))", ac, br)
	}
	res = mat.NewDense(ar, bc, nil)
	res.Mul(a, b)

	return res, nil
}

func EuclideanDistance(p1, p2 mat.Vector) (float64, error) {
	p1len := p1.Len()
	if p1len != p2.Len() {
		return -1, fmt.Errorf("mismatched dimensions")
	}

	total := 0.0
	for i := range p1len {
		diff := p1.AtVec(i) - p2.AtVec(i)
		total += diff * diff
	}

	return math.Sqrt(total), nil
}

// slice 에 동일한 함수가 있으나 편의상 추가
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

// slice 에 동일한 함수가 있으나 편의상 추가
func Transpose(source [][]float64) [][]float64 {
	c := len(source)

	res := make([][]float64, len(source[0]))
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
