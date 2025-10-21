package lapjv

/*
해당 패키지는
Matlab LAPJV,
Python scipy.linear_sum_assignment를 참고하여 작성했습니다.
*/

import (
	"learn/slice/utils"
)

// MaxValue is the maximum cost allowed in the matrix
const MaxValue = 100000.

// NewResult instantiates an allocated Result
func NewResult(dim int) *Result {
	return &Result{
		RowIndex: make([]int, dim),
		ColIndex: make([]int, dim),
	}
}

// Result returns by the LAPJV
type (
	Result struct {
		// Total cost
		Cost float64
		// Assignments index
		RowIndex []int
		ColIndex []int
	}

	LapParam struct {
		matrix [][]float64
		dim    int
		free   []int
		rowsol []int
		colsol []int
		v      []float64

		collist []int
		pred    []int
		d       []float64
	}
)

func Solve(orgMat [][]float64) *Result {
	rowsLen := len(orgMat)
	// no data
	if rowsLen < 1 {
		return nil
	}
	// [1][] float64
	if rowsLen == 1 {
		result := NewResult(rowsLen)
		index, value := utils.MinIndexValue(orgMat[0])
		result.RowIndex[0] = 0
		result.ColIndex[0] = index
		result.Cost = value
		return result
	}
	colsLen := len((orgMat)[0])
	// [][1] float64
	if colsLen == 1 {
		result := NewResult(colsLen)
		trans := utils.Transpose(orgMat)
		index, value := utils.MinIndexValue(trans[0])
		result.RowIndex[0] = index
		result.ColIndex[0] = 0
		result.Cost = value
		return result
	}
	// square 로 계산
	matrix := orgMat
	if rowsLen != colsLen {
		matrix = toSquare(orgMat, rowsLen, colsLen)
	}

	result := NewResult(min(rowsLen, colsLen))

	dim := len(matrix)
	lp := &LapParam{
		dim:    dim,
		matrix: matrix,
		free:   make([]int, dim),
		rowsol: make([]int, dim),
		colsol: make([]int, dim),
		v:      make([]float64, dim),
	}

	// Initilization Phase
	numfree := lp.reduction()

	// Augmenting reduction of unassigned rows
	lp.augmentation(numfree)

	// make result
	// calc cost
	result.Cost = 0.0
	for i := range dim {
		j := lp.rowsol[i]
		result.Cost += matrix[i][j]
	}

	// make index array
	if rowsLen > colsLen {
		idx := 0
		for row := range rowsLen {
			if lp.rowsol[row] < colsLen {
				result.RowIndex[idx] = row
				result.ColIndex[idx] = lp.rowsol[row]
				idx++
			}
		}
	} else {
		for i := range rowsLen {
			result.RowIndex[i] = i
			result.ColIndex[i] = lp.rowsol[i]
		}
	}

	return result
}

func (lp *LapParam) reduction() (numfree int) {
	matches := make([]int, lp.dim)
	// column reduction
	for j := lp.dim - 1; j >= 0; j-- {
		minv := lp.matrix[0][j]
		imin := 0
		for i := 1; i < lp.dim; i++ {
			if lp.matrix[i][j] < minv {
				minv = lp.matrix[i][j]
				imin = i
			}
		}

		lp.v[j] = minv
		matches[imin]++
		if matches[imin] == 1 {
			lp.rowsol[imin] = j
			lp.colsol[j] = imin
		} else {
			lp.colsol[j] = -1
		}
	}

	// Reduction transfer from unassigned to assigned rows
	numfree = 0
	for i := range lp.dim {
		switch matches[i] {
		case 0:
			lp.free[numfree] = i
			numfree++
		case 1:
			j1 := lp.rowsol[i]
			minv := MaxValue
			for j := range lp.dim {
				if j != j1 && lp.matrix[i][j]-lp.v[j] < minv {
					minv = lp.matrix[i][j] - lp.v[j]
				}
			}
			lp.v[j1] -= minv
		}
	}

	var (
		umin, usubmin float64
		j1, j2        int
	)
	// Augmenting reduction of unassigned rows
	for range 2 {
		k := 0
		prvnumfree := numfree
		numfree = 0
		for k < prvnumfree {
			i := lp.free[k]
			k++
			umin = lp.matrix[i][0] - lp.v[0]
			j1 = 0
			usubmin = MaxValue

			for j := 1; j < lp.dim; j++ {
				h := lp.matrix[i][j] - lp.v[j]

				if h < usubmin {
					if h >= umin {
						usubmin = h
						j2 = j
					} else {
						usubmin = umin
						umin = h
						j2 = j1
						j1 = j
					}
				}
			}

			i0 := lp.colsol[j1]
			if umin < usubmin {
				lp.v[j1] -= usubmin - umin
			} else if i0 >= 0 {
				j1 = j2
				i0 = lp.colsol[j2]
			}

			lp.rowsol[i] = j1
			lp.colsol[j1] = i
			if i0 >= 0 {
				if umin < usubmin {
					k--
					lp.free[k] = i0
				} else {
					lp.free[numfree] = i0
					numfree++
				}
			}
		}
	}

	return numfree
}

func (lp *LapParam) augmentation(numfree int) {
	lp.collist = make([]int, lp.dim)
	lp.pred = make([]int, lp.dim)
	lp.d = make([]float64, lp.dim)

	for f := range numfree {
		freerow := lp.free[f]
		for j := range lp.dim {
			lp.d[j] = lp.matrix[freerow][j] - lp.v[j]
			lp.pred[j] = freerow
			lp.collist[j] = j
		}

		endofpath, last, minh := lp.findAugmentingPath()

		for k := 0; k <= last; k++ {
			j1 := lp.collist[k]
			lp.v[j1] += lp.d[j1] - minh
		}

		i := freerow + 1
		for i != freerow {
			i = lp.pred[endofpath]
			lp.colsol[endofpath] = i
			// swap
			endofpath, lp.rowsol[i] = lp.rowsol[i], endofpath
		}
	}
}

func (lp *LapParam) findAugmentingPath() (endofpath, last int, minh float64) {
	low := 0
	up := 0

	for {
		if up == low {
			last = low - 1
			minh = lp.d[lp.collist[up]]
			up++

			for k := up; k < lp.dim; k++ {
				j := lp.collist[k]
				h := lp.d[j]
				if h <= minh {
					if h < minh {
						up = low
						minh = h
					}
					lp.collist[k] = lp.collist[up]
					lp.collist[up] = j
					up++
				}
			}

			for k := low; k < up; k++ {
				if lp.colsol[lp.collist[k]] < 0 {
					endofpath = lp.collist[k]
					return endofpath, last, minh
				}
			}
		}

		j1 := lp.collist[low]
		low++
		i := lp.colsol[j1]
		h := lp.matrix[i][j1] - lp.v[j1] - minh

		for k := up; k < lp.dim; k++ {
			j := lp.collist[k]
			v2 := lp.matrix[i][j] - lp.v[j] - h

			if v2 < lp.d[j] {
				lp.pred[j] = i

				if v2 == minh {
					if lp.colsol[j] < 0 {
						endofpath = j
						return endofpath, last, minh
					}

					lp.collist[k] = lp.collist[up]
					lp.collist[up] = j
					up++
				}

				lp.d[j] = v2
			}
		}
	}
}

// toSquare squarify a matrix
func toSquare(m [][]float64, rl, cl int) [][]float64 {
	size := max(cl, rl)
	matrix := make([][]float64, size)

	for i := range size {
		matrix[i] = make([]float64, size)
		for j := range size {
			if i < rl && j < cl {
				matrix[i][j] = m[i][j]
			} else {
				matrix[i][j] = 0.0
			}
		}
	}

	return matrix
}
